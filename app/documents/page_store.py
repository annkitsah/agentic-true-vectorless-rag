import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from app.documents.models import PageRecord


class PageStoreError(RuntimeError):
    """Base exception for page-store failures."""


class PageNotFoundError(PageStoreError):
    """Raised when a requested page does not exist."""


class PageStore:
    """Persistent filesystem-backed storage for page records."""

    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir

    def _document_dir(self, document_id: str) -> Path:
        self._validate_document_id(document_id)
        return self.root_dir / "documents" / document_id

    def _pages_dir(self, document_id: str) -> Path:
        return self._document_dir(document_id) / "pages"

    def _page_path(
        self,
        document_id: str,
        page_number: int,
    ) -> Path:
        if page_number < 1:
            raise ValueError("page_number must be greater than or equal to 1")

        return self._pages_dir(document_id) / f"{page_number:06d}.json"

    @staticmethod
    def _validate_document_id(document_id: str) -> None:
        if not document_id:
            raise ValueError("document_id cannot be empty")

        if document_id in {".", ".."}:
            raise ValueError("invalid document_id")

        if "/" in document_id or "\\" in document_id:
            raise ValueError("document_id cannot contain path separators")

    def save_page(self, page: PageRecord) -> Path:
        """Persist one page atomically and return its path."""

        pages_dir = self._pages_dir(page.document_id)
        pages_dir.mkdir(parents=True, exist_ok=True)

        destination = self._page_path(
            page.document_id,
            page.page_number,
        )

        payload = page.model_dump(mode="json")

        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=pages_dir,
            prefix=".page-",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)

            try:
                json.dump(
                    payload,
                    temporary_file,
                    ensure_ascii=False,
                    indent=2,
                )
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            except Exception:
                temporary_path.unlink(missing_ok=True)
                raise

        os.replace(temporary_path, destination)

        return destination

    def save_pages(
        self,
        pages: list[PageRecord],
    ) -> list[Path]:
        """Persist multiple pages."""

        if not pages:
            return []

        document_ids = {page.document_id for page in pages}

        if len(document_ids) != 1:
            raise ValueError(
                "All pages in one save operation must belong "
                "to the same document"
            )

        return [self.save_page(page) for page in pages]

    def get_page(
        self,
        document_id: str,
        page_number: int,
    ) -> PageRecord:
        """Load a single page."""

        path = self._page_path(
            document_id,
            page_number,
        )

        if not path.is_file():
            raise PageNotFoundError(
                f"Page {page_number} for document "
                f"{document_id!r} does not exist"
            )

        try:
            payload = json.loads(
                path.read_text(encoding="utf-8")
            )
            return PageRecord.model_validate(payload)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise PageStoreError(
                f"Unable to read page: {path}"
            ) from exc

    def get_pages(
        self,
        document_id: str,
        *,
        start_page: int = 1,
        end_page: int | None = None,
    ) -> list[PageRecord]:
        """Load a page range in ascending page order."""

        if start_page < 1:
            raise ValueError(
                "start_page must be greater than or equal to 1"
            )

        if end_page is not None and end_page < start_page:
            raise ValueError(
                "end_page must be greater than or equal to start_page"
            )

        pages_dir = self._pages_dir(document_id)

        if not pages_dir.is_dir():
            return []

        page_paths = sorted(pages_dir.glob("*.json"))

        pages: list[PageRecord] = []

        for path in page_paths:
            page_number = int(path.stem)

            if page_number < start_page:
                continue

            if end_page is not None and page_number > end_page:
                break

            pages.append(
                self.get_page(
                    document_id,
                    page_number,
                )
            )

        return pages

    def page_exists(
        self,
        document_id: str,
        page_number: int,
    ) -> bool:
        """Return whether a page exists."""

        return self._page_path(
            document_id,
            page_number,
        ).is_file()

    def document_exists(
        self,
        document_id: str,
    ) -> bool:
        """Return whether any persisted pages exist for a document."""

        return self._pages_dir(document_id).is_dir()

    def delete_document(
        self,
        document_id: str,
    ) -> None:
        """Delete all persisted pages for a document."""

        document_dir = self._document_dir(document_id)

        if not document_dir.exists():
            return

        for path in sorted(
            document_dir.rglob("*"),
            reverse=True,
        ):
            if path.is_file() or path.is_symlink():
                path.unlink()
            elif path.is_dir():
                path.rmdir()

        document_dir.rmdir()