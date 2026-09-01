from app.documents.models import PageRecord
from app.documents.page_store import PageStore
from app.retrieval.page_index import PageIndex


class IndexLifecycle:
    """Manage synchronization between persisted pages and the search index."""

    def __init__(
        self,
        page_store: PageStore,
        *,
        page_index: PageIndex | None = None,
    ) -> None:
        self.page_store = page_store
        self.page_index = page_index or PageIndex(
            page_store=page_store,
        )

    def build(self) -> int:
        """Index all currently persisted pages without clearing the index."""

        indexed_count = 0

        for document_id in self.page_store.get_document_ids():
            indexed_count += self.index_document(document_id)

        return indexed_count

    def rebuild(self) -> int:
        """Clear the current index and rebuild it from persisted pages."""

        self.clear()

        return self.build()

    def index_page(
        self,
        page: PageRecord,
    ) -> None:
        """Add or replace a single page in the search index."""

        self.page_index.add_page(page)

    def index_document(
        self,
        document_id: str,
    ) -> int:
        """Index every persisted page belonging to a document."""

        if not document_id.strip():
            raise ValueError("document_id cannot be empty")

        pages = self.page_store.get_pages(document_id)

        for page in pages:
            self.index_page(page)

        return len(pages)

    def remove_page(
        self,
        page_id: str,
    ) -> None:
        """Remove a single page from the search index."""

        self.page_index.remove_page(page_id)

    def remove_document(
        self,
        document_id: str,
    ) -> int:
        """Remove every persisted page of a document from the search index."""

        if not document_id.strip():
            raise ValueError("document_id cannot be empty")

        pages = self.page_store.get_pages(document_id)

        for page in pages:
            self.remove_page(page.page_id)

        return len(pages)

    def clear(self) -> None:
        """Remove every page from the search index."""

        self.page_index.clear()