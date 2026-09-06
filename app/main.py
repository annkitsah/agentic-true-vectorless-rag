from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config.settings import get_settings
from app.container import ApplicationContainer, create_application_container

settings = get_settings()

container: ApplicationContainer = create_application_container(settings)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Prepare persistent state before the app starts serving requests."""

    app.state.indexed_page_count = container.initialize()

    yield


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    lifespan=lifespan,
)

app.state.container = container


@app.get("/health")
async def health_check() -> dict[str, str | int]:
    return {
        "status": "healthy",
        "service": settings.app_name,
        "environment": settings.app_env,
        "indexed_pages": getattr(app.state, "indexed_page_count", 0),
    }