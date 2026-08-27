from pathlib import Path

import pytest

from app.documents.models import PageRecord
from app.documents.page_store import PageStore
from app.retrieval.lexical import LexicalRetriever
from app.retrieval.models import RetrievalQuery


def create_page(
    document_id: str,
    page_number: int,
    text: str,
) -> PageRecord:
    return PageRecord(
        document_id=document_id,
        page_number=page_number,
        text=text,
        width=612.0,
        height=792.0,
    )


@pytest.fixture
def page_store(tmp_path: Path) -> PageStore:
    store = PageStore(tmp_path)

    store.save_pages(
        [
            create_page(
                document_id="doc-001",
                page_number=1,
                text=(
                    "Retrieval augmented generation combines "
                    "retrieval with language models."
                ),
            ),
            create_page(
                document_id="doc-001",
                page_number=2,
                text=(
                    "Vectorless retrieval uses lexical matching "
                    "and traditional information retrieval."
                ),
            ),
            create_page(
                document_id="doc-001",
                page_number=3,
                text=(
                    "The weather today is sunny and warm."
                ),
            ),
        ]
    )

    store.save_pages(
        [
            create_page(
                document_id="doc-002",
                page_number=1,
                text=(
                    "BM25 is a lexical ranking algorithm "
                    "used in information retrieval."
                ),
            ),
        ]
    )

    return store


def test_retriever_returns_relevant_pages(
    page_store: PageStore,
) -> None:
    retriever = LexicalRetriever(page_store)

    response = retriever.retrieve(
        RetrievalQuery(
            text="lexical retrieval",
            top_k=2,
        )
    )

    assert response.result_count == 2

    assert response.results[0].document_id == "doc-001"
    assert response.results[0].page_number == 2

    assert "lexical" in response.results[0].matched_terms
    assert "retrieval" in response.results[0].matched_terms


def test_retriever_respects_top_k(
    page_store: PageStore,
) -> None:
    retriever = LexicalRetriever(page_store)

    response = retriever.retrieve(
        RetrievalQuery(
            text="retrieval",
            top_k=1,
        )
    )

    assert response.result_count == 1


def test_retriever_can_filter_by_document(
    page_store: PageStore,
) -> None:
    retriever = LexicalRetriever(page_store)

    response = retriever.retrieve(
        RetrievalQuery(
            text="information retrieval",
            document_id="doc-002",
            top_k=10,
        )
    )

    assert response.result_count == 1

    result = response.results[0]

    assert result.document_id == "doc-002"
    assert result.page_number == 1


def test_retriever_does_not_return_unmatched_pages(
    page_store: PageStore,
) -> None:
    retriever = LexicalRetriever(page_store)

    response = retriever.retrieve(
        RetrievalQuery(
            text="quantum computing",
            top_k=10,
        )
    )

    assert response.results == ()


def test_retriever_normalizes_query(
    page_store: PageStore,
) -> None:
    retriever = LexicalRetriever(page_store)

    response = retriever.retrieve(
        RetrievalQuery(
            text="  LEXICAL   RETRIEVAL  ",
            top_k=5,
        )
    )

    assert response.result_count > 0

    assert response.results[0].page_number == 2


def test_retriever_ignores_stopwords(
    page_store: PageStore,
) -> None:
    retriever = LexicalRetriever(page_store)

    response = retriever.retrieve(
        RetrievalQuery(
            text="the lexical retrieval",
            top_k=5,
        )
    )

    assert response.result_count > 0
    assert response.results[0].page_number == 2


def test_retriever_returns_deterministic_order(
    page_store: PageStore,
) -> None:
    retriever = LexicalRetriever(page_store)

    query = RetrievalQuery(
        text="retrieval",
        top_k=10,
    )

    first = retriever.retrieve(query)
    second = retriever.retrieve(query)

    assert first == second


def test_retriever_rejects_invalid_bm25_parameters(
    page_store: PageStore,
) -> None:
    with pytest.raises(ValueError):
        LexicalRetriever(
            page_store,
            k1=-1,
        )

    with pytest.raises(ValueError):
        LexicalRetriever(
            page_store,
            b=1.5,
        )


def test_retriever_empty_document_store(
    tmp_path: Path,
) -> None:
    store = PageStore(tmp_path)
    retriever = LexicalRetriever(store)

    response = retriever.retrieve(
        RetrievalQuery(
            text="retrieval",
        )
    )

    assert response.results == ()