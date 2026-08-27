from pathlib import Path

import pytest

from app.documents.models import DocumentStatus
from app.documents.page_store import PageStore
from app.documents.repository import DocumentRepository
from app.ingestion.service import IngestionService


@pytest.fixture
def repository(tmp_path: Path) -> DocumentRepository:
    database_path = tmp_path / "documents.db"

    repository = DocumentRepository(database_path)
    repository.initialize()

    return repository


@pytest.fixture
def page_store(tmp_path: Path) -> PageStore:
    return PageStore(
        tmp_path / "processed",
    )


@pytest.fixture
def service(
    repository: DocumentRepository,
    page_store: PageStore,
) -> IngestionService:
    return IngestionService(
        repository=repository,
        page_store=page_store,
    )


def test_ingest_pdf(
    service: IngestionService,
    page_store: PageStore,
) -> None:
    pdf_path = Path(
        "data/raw/Chapter 1 - The Overview of Map of GenAI.pdf"
    )

    result = service.ingest(pdf_path)

    assert result.duplicate is False
    assert result.document.status == DocumentStatus.PROCESSED
    assert result.document.filename == pdf_path.name
    assert result.page_count == result.document.page_count
    assert result.page_count > 0

    first_page = page_store.get_page(
        result.document.document_id,
        1,
    )

    assert first_page.document_id == result.document.document_id
    assert first_page.page_number == 1
    assert len(first_page.text) > 0


def test_all_pages_are_persisted(
    service: IngestionService,
    page_store: PageStore,
) -> None:
    pdf_path = Path(
        "data/raw/Chapter 1 - The Overview of Map of GenAI.pdf"
    )

    result = service.ingest(pdf_path)

    pages = page_store.get_pages(
        result.document.document_id,
    )

    assert len(pages) == result.page_count
    assert [page.page_number for page in pages] == list(
        range(1, result.page_count + 1)
    )


def test_ingest_same_pdf_twice_is_idempotent(
    service: IngestionService,
    page_store: PageStore,
) -> None:
    pdf_path = Path(
        "data/raw/Chapter 1 - The Overview of Map of GenAI.pdf"
    )

    first_result = service.ingest(pdf_path)
    second_result = service.ingest(pdf_path)

    assert first_result.duplicate is False
    assert second_result.duplicate is True

    assert (
        first_result.document.document_id
        == second_result.document.document_id
    )

    assert (
        first_result.document.file_hash
        == second_result.document.file_hash
    )

    pages = page_store.get_pages(
        second_result.document.document_id,
    )

    assert len(pages) == second_result.page_count


def test_duplicate_ingestion_does_not_rewrite_pages(
    service: IngestionService,
    page_store: PageStore,
) -> None:
    pdf_path = Path(
        "data/raw/Chapter 1 - The Overview of Map of GenAI.pdf"
    )

    first_result = service.ingest(pdf_path)

    page_path = page_store._page_path(
        first_result.document.document_id,
        1,
    )

    original_mtime = page_path.stat().st_mtime_ns

    second_result = service.ingest(pdf_path)

    assert second_result.duplicate is True

    current_mtime = page_path.stat().st_mtime_ns

    assert current_mtime == original_mtime


def test_ingest_missing_file(
    service: IngestionService,
) -> None:
    with pytest.raises(FileNotFoundError):
        service.ingest(
            Path("data/raw/missing.pdf"),
        )


def test_ingest_non_pdf(
    service: IngestionService,
    tmp_path: Path,
) -> None:
    text_file = tmp_path / "document.txt"
    text_file.write_text("not a PDF")

    with pytest.raises(ValueError):
        service.ingest(text_file)