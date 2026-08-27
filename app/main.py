from fastapi import FastAPI

from app.config.settings import get_settings
from app.container import create_application_container


settings = get_settings()

container = create_application_container()

ingestion_service = container.ingestion_service


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
)


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {
        "status": "healthy",
        "service": settings.app_name,
        "environment": settings.app_env,
    }