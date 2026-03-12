from functools import lru_cache

from app.config import Settings, settings
from app.services.task_manager import TaskManager


@lru_cache()
def get_settings() -> Settings:
    return settings


_task_manager: TaskManager | None = None


def get_task_manager() -> TaskManager:
    global _task_manager
    if _task_manager is None:
        _task_manager = TaskManager(max_workers=settings.max_workers)
    return _task_manager
