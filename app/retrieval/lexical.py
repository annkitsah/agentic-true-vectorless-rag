from collections.abc import Iterable

from app.documents.models import PageRecord
from app.documents.page_store import PageStore
from app.retrieval.models import (
    RetrievalQuery,
    RetrievalResponse,
    RetrievalResult,
)
from app.retrieval.text import term_frequency, tokenize


class LexicalRetriever:
    """Deterministic lexical retriever over persisted document pages."""

    def __init__(self, page_store: PageStore, *, k1: float = 1.2, b: float = 0.75) -> None:
        if k1 <= 0:
            raise ValueError("k1 must be greater than zero")

        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1")
        self.page_store = page_store
        self.k1 = k1
        self.b = b

    def retrieve(
        self,
        query: RetrievalQuery,
    ) -> RetrievalResponse:
        query_tokens = tokenize(query.text)
        query_frequencies = term_frequency(query_tokens)

        if not query_frequencies:
            return RetrievalResponse(
                query=query,
                results=(),
            )

        pages = self._load_pages(query.document_id)

        scored: list[RetrievalResult] = []

        for page in pages:
            page_tokens = tokenize(page.text)
            page_frequencies = term_frequency(page_tokens)

            matched_terms = tuple(
                sorted(
                    term
                    for term in query_frequencies
                    if term in page_frequencies
                )
            )

            if not matched_terms:
                continue

            score = self._score(
                query_frequencies=query_frequencies,
                page_frequencies=page_frequencies,
            )

            scored.append(
                RetrievalResult(
                    document_id=page.document_id,
                    page_number=page.page_number,
                    text=page.text,
                    score=score,
                    matched_terms=matched_terms,
                )
            )

        scored.sort(
            key=lambda result: (
                -result.score,
                result.document_id,
                result.page_number,
            )
        )

        return RetrievalResponse(
            query=query,
            results=tuple(scored[: query.top_k]),
        )

    def _load_pages(
        self,
        document_id: str | None,
    ) -> Iterable[PageRecord]:
        if document_id is not None:
            return self.page_store.get_pages(document_id)

        return self._load_all_pages()

    def _load_all_pages(self) -> list[PageRecord]:
        """
        Load all persisted pages.

        This method should use the PageStore's existing document/page
        discovery API. Do not access the filesystem directly here.
        """
        return self.page_store.get_all_pages()

    @staticmethod
    def _score(
        *,
        query_frequencies: dict[str, int],
        page_frequencies: dict[str, int],
    ) -> float:
        """Calculate a simple deterministic lexical relevance score."""

        score = 0.0

        for term, query_frequency in query_frequencies.items():
            page_frequency = page_frequencies.get(term, 0)

            if page_frequency == 0:
                continue

            score += float(query_frequency * page_frequency)

        return score