from pathlib import Path

from app.config.settings import Settings, get_settings
from app.documents.page_store import PageStore
from app.documents.repository import DocumentRepository
from app.ingestion.service import IngestionService
from app.ocr.pipeline import OCRPipeline
from app.ocr.providers.mistral import MistralOCRProvider


class ApplicationContainer:
    """Application dependency composition root."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

        self.repository = DocumentRepository(
            Path(settings.metadata_dir),
        )

        self.page_store = PageStore(
            Path(settings.processed_data_dir),
        )

        self.ocr_pipeline = self._build_ocr_pipeline()

        self.ingestion_service = IngestionService(
            repository=self.repository,
            page_store=self.page_store,
            ocr_pipeline=self.ocr_pipeline,
            processed_root=Path(settings.processed_data_dir),
        )

    def _build_ocr_pipeline(self) -> OCRPipeline:
        if not self.settings.ocr_enabled:
            raise RuntimeError(
                "OCR is disabled, but no alternative OCR pipeline "
                "implementation is configured."
            )

        if not self.settings.mistral_api_key:
            raise RuntimeError(
                "MISTRAL_API_KEY is required when OCR is enabled."
            )

        provider = MistralOCRProvider(
            api_key=self.settings.mistral_api_key,
            model=self.settings.mistral_ocr_model,
            timeout_ms=self.settings.mistral_ocr_timeout_ms,
        )

        return OCRPipeline(
            provider=provider,
        )


def create_application_container(
    settings: Settings | None = None,
) -> ApplicationContainer:
    """Build the application's dependency graph."""

    resolved_settings = settings or get_settings()

    return ApplicationContainer(
        settings=resolved_settings,
    )