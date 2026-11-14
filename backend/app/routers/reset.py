"""Reset endpoints: preview what is stored, then delete it all."""

import ipaddress
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.concurrency import run_in_threadpool

from app.dependencies import get_task_manager
from app.services import reset
from app.services.task_manager import TaskManager

router = APIRouter()

_LOCAL_SUFFIXES = (".local", ".lan", ".home")


def _is_local_origin(origin: str) -> bool:
    """True for a page served from this machine or the local network."""
    host = urlsplit(origin).hostname or ""
    if host == "localhost" or "." not in host and ":" not in host and host:
        return True  # localhost, or a bare machine name on the LAN
    if host.endswith(_LOCAL_SUFFIXES):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return ip.is_loopback or ip.is_private


def _refuse_foreign_origin(request: Request) -> None:
    """Stop a web page on the public internet from wiping the data.

    The API allows cross-origin calls, so without this any site open in the
    browser could post here. A request with no Origin header (curl, a script)
    is allowed. This is a guard against drive-by requests, not authentication.
    """
    origin = request.headers.get("origin")
    if origin and not _is_local_origin(origin):
        raise HTTPException(403, "Reset is only allowed from this machine or the local network")


@router.get("/reset/preview")
async def reset_preview():
    """Counts of what a reset would delete. Deletes nothing."""
    return await run_in_threadpool(reset.preview)


@router.post("/reset")
async def reset_everything(request: Request, tm: TaskManager = Depends(get_task_manager)):
    """Delete every project (photos, music, caches, videos) and the task history."""
    _refuse_foreign_origin(request)
    if tm.has_running():
        raise HTTPException(409, "A task is still running. Cancel it or wait for it to finish, then reset.")

    result = await run_in_threadpool(reset.reset_all)
    tm.forget_finished()
    return result
