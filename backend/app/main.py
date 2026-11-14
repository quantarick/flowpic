import io
import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI

# Force UTF-8 logging on Windows to handle Māori/CJK place names
_handler = logging.StreamHandler(
    io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
)
_handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
logging.basicConfig(level=logging.INFO, handlers=[_handler])

from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.dependencies import get_task_manager
from app.routers import upload, project, generate, download, llm, ollama, reset, styles, copywriting, xhs_publish


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: ensure data dir exists
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    # No task survives a restart, so anything still marked as running is dead.
    from app.services.task_db import fail_unfinished
    interrupted = fail_unfinished("Interrupted: the server stopped while this task was running")
    if interrupted:
        logging.getLogger(__name__).info("Marked %d interrupted task(s) as failed", interrupted)
    yield
    # Shutdown: clean up task manager
    tm = get_task_manager()
    tm.shutdown()


app = FastAPI(
    title="FlowPic",
    description="Music-driven photo slideshow video generator",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router, prefix="/api/upload", tags=["upload"])
app.include_router(project.router, prefix="/api/project", tags=["project"])
app.include_router(generate.router, prefix="/api", tags=["generate"])
app.include_router(generate.ws_router, prefix="/ws", tags=["websocket"])
app.include_router(download.router, prefix="/api", tags=["download"])
app.include_router(llm.router, prefix="/api/llm", tags=["llm"])
app.include_router(ollama.router, prefix="/api/ollama", tags=["ollama"])
app.include_router(styles.router, prefix="/api", tags=["styles"])
app.include_router(reset.router, prefix="/api", tags=["reset"])
app.include_router(copywriting.router, prefix="/api", tags=["copywriting"])
app.include_router(xhs_publish.router, prefix="/api", tags=["xhs"])


@app.get("/api/health")
async def health():
    return {"status": "ok"}
