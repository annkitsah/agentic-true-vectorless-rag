import math
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
    """Deterministic BM25 lexical retriever over persisted document pages."""

    def __init__(
        self,
        page_store: PageStore,
        *,
        k1: float = 1.2,
        b: float = 0.75,
    ) -> None:
        if not math.isfinite(k1) or k1 <= 0:
            raise ValueError("k1 must be greater than zero")

        if not math.isfinite(b) or not 0 <= b <= 1:
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

        pages = list(self._load_pages(query.document_id))

        if not pages:
            return RetrievalResponse(
                query=query,
                results=(),
            )

        page_data = [
            (
                page,
                tokenize(page.text),
            )
            for page in pages
        ]

        page_frequencies = [
            term_frequency(tokens)
            for _, tokens in page_data
        ]

        document_count = len(page_frequencies)

        average_document_length = (
            sum(len(tokens) for _, tokens in page_data)
            / document_count
        )

        document_frequencies = self._document_frequencies(
            page_frequencies,
        )

        scored: list[RetrievalResult] = []

        for page, frequencies in zip(
            (page for page, _ in page_data),
            page_frequencies,
            strict=True,
        ):
            matched_terms = tuple(
                sorted(
                    term
                    for term in query_frequencies
                    if term in frequencies
                )
            )

            if not matched_terms:
                continue

            score = self._score(
                query_frequencies=query_frequencies,
                page_frequencies=frequencies,
                document_frequencies=document_frequencies,
                document_count=document_count,
                document_length=sum(frequencies.values()),
                average_document_length=average_document_length,
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
        """Load all persisted pages through PageStore."""

        return self.page_store.get_all_pages()

    @staticmethod
    def _document_frequencies(
        page_frequencies: list[dict[str, int]],
    ) -> dict[str, int]:
        """Count the number of pages containing each term."""

        frequencies: dict[str, int] = {}

        for page_frequency in page_frequencies:
            for term in page_frequency:
                frequencies[term] = frequencies.get(term, 0) + 1

        return frequencies

    def _score(
        self,
        *,
        query_frequencies: dict[str, int],
        page_frequencies: dict[str, int],
        document_frequencies: dict[str, int],
        document_count: int,
        document_length: int,
        average_document_length: float,
    ) -> float:
        """Calculate the BM25 relevance score for one page."""

        if document_count <= 0:
            return 0.0

        if average_document_length <= 0:
            return 0.0

        score = 0.0

        length_normalization = (
            1.0
            - self.b
            + self.b
            * (document_length / average_document_length)
        )

        for term, query_frequency in query_frequencies.items():
            term_frequency_value = page_frequencies.get(term, 0)

            if term_frequency_value == 0:
                continue

            document_frequency = document_frequencies.get(term, 0)

            idf = math.log(
                1.0
                + (
                    (document_count - document_frequency + 0.5)
                    / (document_frequency + 0.5)
                )
            )

            term_score = (
                idf
                * (
                    (
                        term_frequency_value
                        * (self.k1 + 1.0)
                    )
                    / (
                        term_frequency_value
                        + self.k1
                        * length_normalization
                    )
                )
            )

            score += query_frequency * term_score

        return score