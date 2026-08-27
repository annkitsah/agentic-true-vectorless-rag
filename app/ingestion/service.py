from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, ConfigDict

from app.documents.models import DocumentRecord, DocumentStatus
from app.documents.page_store import PageStore
from app.documents.repository import DocumentRepository
from app.ingestion.hashing import calculate_file_sha256
from app.ingestion.pdf_parser import extract_pages


class IngestionResult(BaseModel):
    """Result returned after processing a document."""

    model_config = ConfigDict(frozen=True)

    document: DocumentRecord
    page_count: int
    duplicate: bool


class IngestionService:
    """Coordinates document registration and persistent page extraction."""

    def __init__(
        self,
        repository: DocumentRepository,
        page_store: PageStore,
    ) -> None:
        self.repository = repository
        self.page_store = page_store

    def ingest(self, file_path: Path) -> IngestionResult:
        """Ingest a PDF into the document registry and page store."""

        if not file_path.is_file():
            raise FileNotFoundError(
                f"Document does not exist: {file_path}"
            )

        if file_path.suffix.lower() != ".pdf":
            raise ValueError(
                f"Unsupported document type: {file_path.suffix}"
            )

        file_hash = calculate_file_sha256(file_path)

        existing_document = self.repository.get_by_hash(file_hash)

        if existing_document is not None:
            return IngestionResult(
                document=existing_document,
                page_count=existing_document.page_count,
                duplicate=True,
            )

        document_id = f"doc_{uuid4().hex}"

        pages = extract_pages(
            file_path,
            document_id,
        )

        self.page_store.save_pages(pages)

        document = DocumentRecord(
            document_id=document_id,
            filename=file_path.name,
            source_path=str(file_path.resolve()),
            file_hash=file_hash,
            file_size_bytes=file_path.stat().st_size,
            page_count=len(pages),
            status=DocumentStatus.PROCESSED,
        )

        self.repository.add(document)

        return IngestionResult(
            document=document,
            page_count=len(pages),
            duplicate=False,
        )