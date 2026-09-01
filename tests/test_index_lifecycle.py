from pathlib import Path

import pytest

from app.documents.models import PageRecord
from app.documents.page_store import PageStore
from app.retrieval.index_lifecycle import IndexLifecycle
from app.retrieval.page_index import PageIndex


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
                "doc-001",
                1,
                "Python machine learning systems.",
            ),
            create_page(
                "doc-001",
                2,
                "Retrieval augmented generation.",
            ),
        ]
    )

    store.save_pages(
        [
            create_page(
                "doc-002",
                1,
                "Database indexing systems.",
            ),
        ]
    )

    return store


def test_build_indexes_all_persisted_pages(
    page_store: PageStore,
) -> None:
    lifecycle = IndexLifecycle(page_store)

    count = lifecycle.build()

    assert count == 3
    assert lifecycle.page_index.lookup("python") == (
        "doc-001:page:1",
    )
    assert lifecycle.page_index.lookup("retrieval") == (
        "doc-001:page:2",
    )
    assert lifecycle.page_index.lookup("database") == (
        "doc-002:page:1",
    )


def test_build_is_idempotent(
    page_store: PageStore,
) -> None:
    lifecycle = IndexLifecycle(page_store)

    first_count = lifecycle.build()
    first_lookup = lifecycle.page_index.lookup("retrieval")

    second_count = lifecycle.build()
    second_lookup = lifecycle.page_index.lookup("retrieval")

    assert first_count == 3
    assert second_count == 3
    assert first_lookup == second_lookup == (
        "doc-001:page:2",
    )


def test_rebuild_removes_stale_index_entries(
    page_store: PageStore,
) -> None:
    lifecycle = IndexLifecycle(page_store)

    lifecycle.build()

    page_store.save_page(
        create_page(
            "doc-001",
            1,
            "database architecture",
        )
    )

    count = lifecycle.rebuild()

    assert count == 3
    assert lifecycle.page_index.lookup("python") == ()
    assert lifecycle.page_index.lookup("machine") == ()
    assert lifecycle.page_index.lookup("database") == (
        "doc-001:page:1",
        "doc-002:page:1",
    )


def test_index_page_adds_page_to_index(
    page_store: PageStore,
) -> None:
    lifecycle = IndexLifecycle(page_store)

    page = create_page(
        "doc-003",
        1,
        "agentic vectorless retrieval",
    )

    page_store.save_page(page)

    lifecycle.index_page(page)

    assert lifecycle.page_index.lookup("agentic") == (
        "doc-003:page:1",
    )
    assert lifecycle.page_index.lookup("vectorless") == (
        "doc-003:page:1",
    )
    assert lifecycle.page_index.lookup("retrieval") == (
        "doc-003:page:1",
    )


def test_index_page_replaces_existing_page_terms(
    page_store: PageStore,
) -> None:
    lifecycle = IndexLifecycle(page_store)

    original_page = page_store.get_page(
        "doc-001",
        1,
    )

    lifecycle.index_page(original_page)

    updated_page = create_page(
        "doc-001",
        1,
        "database architecture",
    )

    page_store.save_page(updated_page)
    lifecycle.index_page(updated_page)

    assert lifecycle.page_index.lookup("python") == ()
    assert lifecycle.page_index.lookup("machine") == ()
    assert lifecycle.page_index.lookup("database") == (
        "doc-001:page:1",
    )


def test_index_document_indexes_all_document_pages(
    page_store: PageStore,
) -> None:
    lifecycle = IndexLifecycle(page_store)

    count = lifecycle.index_document("doc-001")

    assert count == 2
    assert lifecycle.page_index.lookup("python") == (
        "doc-001:page:1",
    )
    assert lifecycle.page_index.lookup("retrieval") == (
        "doc-001:page:2",
    )


def test_index_document_returns_zero_for_missing_document(
    page_store: PageStore,
) -> None:
    lifecycle = IndexLifecycle(page_store)

    count = lifecycle.index_document("missing-document")

    assert count == 0


def test_remove_page_removes_page_from_index(
    page_store: PageStore,
) -> None:
    lifecycle = IndexLifecycle(page_store)

    lifecycle.build()

    lifecycle.remove_page(
        "doc-001:page:1",
    )

    assert lifecycle.page_index.lookup("python") == ()
    assert lifecycle.page_index.lookup("machine") == ()

    assert lifecycle.page_index.lookup("retrieval") == (
        "doc-001:page:2",
    )


def test_remove_document_removes_only_that_document(
    page_store: PageStore,
) -> None:
    lifecycle = IndexLifecycle(page_store)

    lifecycle.build()

    count = lifecycle.remove_document("doc-001")

    assert count == 2
    assert lifecycle.page_index.lookup("python") == ()
    assert lifecycle.page_index.lookup("retrieval") == ()

    assert lifecycle.page_index.lookup("database") == (
        "doc-002:page:1",
    )


def test_remove_document_returns_zero_for_missing_document(
    page_store: PageStore,
) -> None:
    lifecycle = IndexLifecycle(page_store)

    lifecycle.build()

    count = lifecycle.remove_document(
        "missing-document",
    )

    assert count == 0
    assert lifecycle.page_index.lookup("python") == (
        "doc-001:page:1",
    )


def test_clear_removes_everything_from_index(
    page_store: PageStore,
) -> None:
    lifecycle = IndexLifecycle(page_store)

    lifecycle.build()
    lifecycle.clear()

    assert lifecycle.page_index.lookup("python") == ()
    assert lifecycle.page_index.lookup("retrieval") == ()
    assert lifecycle.page_index.lookup("database") == ()


def test_rebuild_empty_store(
    tmp_path: Path,
) -> None:
    store = PageStore(tmp_path)
    lifecycle = IndexLifecycle(store)

    count = lifecycle.rebuild()

    assert count == 0


def test_index_document_rejects_empty_document_id(
    page_store: PageStore,
) -> None:
    lifecycle = IndexLifecycle(page_store)

    with pytest.raises(ValueError, match="document_id cannot be empty"):
        lifecycle.index_document("   ")


def test_remove_document_rejects_empty_document_id(
    page_store: PageStore,
) -> None:
    lifecycle = IndexLifecycle(page_store)

    with pytest.raises(ValueError, match="document_id cannot be empty"):
        lifecycle.remove_document("   ")


def test_lifecycle_accepts_custom_page_index(
    page_store: PageStore,
) -> None:
    page_index = PageIndex(
        page_store=page_store,
    )

    lifecycle = IndexLifecycle(
        page_store,
        page_index=page_index,
    )

    assert lifecycle.page_index is page_index


def test_rebuild_can_recover_from_stale_index(
    page_store: PageStore,
) -> None:
    lifecycle = IndexLifecycle(page_store)

    lifecycle.build()

    lifecycle.page_index.add_page(
        create_page(
            "stale-document",
            1,
            "stale indexed content",
        )
    )

    assert lifecycle.page_index.lookup("stale") == (
        "stale-document:page:1",
    )

    lifecycle.rebuild()

    assert lifecycle.page_index.lookup("stale") == ()