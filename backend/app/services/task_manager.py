"""Task manager: ThreadPoolExecutor + WebSocket notification."""

import asyncio
import logging
import threading
import time
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
            from app.services.task_db import update_task_status
            # Check cancellation before starting
            if task_id in self._cancelled:
                update_task_status(task_id, TaskStatus.CANCELLED,
                                   error_message="Cancelled before start")
                self._update_progress(task_id, ProgressMessage(
                    status=TaskStatus.CANCELLED, progress=0,
                    current_step="Cancelled", detail="Cancelled before start",
                ))
                return None
            t0 = time.time()
            try:
                result = fn(
                    progress_callback=lambda msg: self._update_progress(task_id, msg),
                    cancel_check=lambda: task_id in self._cancelled,
                    **kwargs,
                )
                elapsed = time.time() - t0
                update_task_status(
                    task_id,
                    TaskStatus.DONE,
                    output_path=result,
                    duration_seconds=round(elapsed, 2),
                )
                return result
            except Exception as e:
                elapsed = time.time() - t0
                logger.exception(f"Task {task_id} failed")
                status = TaskStatus.CANCELLED if task_id in self._cancelled else TaskStatus.FAILED
                update_task_status(task_id, status, duration_seconds=round(elapsed, 2),
                                   error_message=str(e)[:500])
                self._update_progress(
                    task_id,
                    ProgressMessage(
                        status=status,
                        progress=0,
                        current_step="Failed" if status == TaskStatus.FAILED else "Cancelled",
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
        """Cancel a task — works for both queued and running tasks."""
        self._cancelled.add(task_id)

        # Try to cancel the Future (works if still queued, no-op if running)
        future = self._futures.get(task_id)
        if future and future.cancel():
            # Successfully pulled from queue before it started
            from app.services.task_db import update_task_status
            update_task_status(task_id, TaskStatus.CANCELLED,
                               error_message="Cancelled before start")
            self._update_progress(task_id, ProgressMessage(
                status=TaskStatus.CANCELLED, progress=0,
                current_step="Cancelled", detail="Cancelled before start",
            ))

    def get_status(self, task_id: str) -> ProgressMessage | None:
        return self._progress.get(task_id)

    def shutdown(self):
        self._executor.shutdown(wait=False)
