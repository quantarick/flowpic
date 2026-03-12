"""Task manager: ThreadPoolExecutor + WebSocket notification."""

import asyncio
import logging
import threading
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, AsyncGenerator, Callable

from app.models import ProgressMessage, TaskStatus

logger = logging.getLogger(__name__)


class TaskManager:
    def __init__(self, max_workers: int = 2):
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._futures: dict[str, Future] = {}
        self._progress: dict[str, ProgressMessage] = {}
        self._queues: dict[str, list[asyncio.Queue]] = defaultdict(list)
        self._loops: dict[str, list[asyncio.AbstractEventLoop]] = defaultdict(list)
        self._cancelled: set[str] = set()
        self._lock = threading.Lock()

    def submit(
        self,
        task_id: str,
        fn: Callable,
        **kwargs: Any,
    ):
        """Submit a task to the thread pool."""
        self._progress[task_id] = ProgressMessage(
            status=TaskStatus.PENDING, progress=0, current_step="Queued"
        )

        def _wrapper():
            try:
                return fn(
                    progress_callback=lambda msg: self._update_progress(task_id, msg),
                    cancel_check=lambda: task_id in self._cancelled,
                    **kwargs,
                )
            except Exception as e:
                logger.exception(f"Task {task_id} failed")
                self._update_progress(
                    task_id,
                    ProgressMessage(
                        status=TaskStatus.FAILED,
                        progress=0,
                        current_step="Failed",
                        detail=str(e),
                    ),
                )

        future = self._executor.submit(_wrapper)
        self._futures[task_id] = future

    def _update_progress(self, task_id: str, msg: ProgressMessage):
        """Update progress and notify all subscribers (thread-safe)."""
        self._progress[task_id] = msg
        data = msg.model_dump()
        with self._lock:
            queues = list(self._queues.get(task_id, []))
            loops = list(self._loops.get(task_id, []))
        for q, loop in zip(queues, loops):
            try:
                loop.call_soon_threadsafe(q.put_nowait, data)
            except (RuntimeError, asyncio.QueueFull):
                pass

    async def subscribe(self, task_id: str) -> AsyncGenerator[dict, None]:
        """Subscribe to progress updates for a task."""
        q: asyncio.Queue = asyncio.Queue(maxsize=100)
        loop = asyncio.get_event_loop()
        with self._lock:
            self._queues[task_id].append(q)
            self._loops[task_id].append(loop)

        terminal = {"done", "failed", "cancelled"}

        # Send current status immediately
        if task_id in self._progress:
            data = self._progress[task_id].model_dump()
            yield data
            if data.get("status") in terminal:
                return

        while True:
            try:
                msg = await asyncio.wait_for(q.get(), timeout=5.0)
                yield msg
                if msg.get("status") in terminal:
                    break
            except asyncio.TimeoutError:
                # Send current status as keepalive
                if task_id in self._progress:
                    current = self._progress[task_id].model_dump()
                    yield current
                    if current.get("status") in terminal:
                        break

    def unsubscribe(self, task_id: str):
        """Clean up queues for a task."""
        with self._lock:
            self._queues.pop(task_id, None)
            self._loops.pop(task_id, None)

    def cancel(self, task_id: str):
        """Mark a task for cancellation."""
        self._cancelled.add(task_id)

    def get_status(self, task_id: str) -> ProgressMessage | None:
        return self._progress.get(task_id)

    def shutdown(self):
        self._executor.shutdown(wait=False)
