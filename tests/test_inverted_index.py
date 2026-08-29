
import pytest

from app.retrieval.inverted_index import InvertedIndex


def test_add_and_lookup_term() -> None:
    index = InvertedIndex()

    index.add("doc-001:page:1", "retrieval systems")

    assert index.lookup("retrieval") == ("doc-001:page:1",)


def test_index_tokenizes_text() -> None:
    index = InvertedIndex()

    index.add("doc-001:page:1", "Retrieval SYSTEMS")

    assert index.lookup("retrieval") == ("doc-001:page:1",)
    assert index.lookup("systems") == ("doc-001:page:1",)


def test_multiple_pages_for_same_term() -> None:
    index = InvertedIndex()

    index.add("doc-001:page:1", "retrieval system")
    index.add("doc-001:page:2", "retrieval architecture")
    index.add("doc-002:page:1", "retrieval pipeline")

    assert index.lookup("retrieval") == (
        "doc-001:page:1",
        "doc-001:page:2",
        "doc-002:page:1",
    )


def test_unknown_term_returns_empty_tuple() -> None:
    index = InvertedIndex()

    index.add("doc-001:page:1", "retrieval system")

    assert index.lookup("unknown") == ()


def test_lookup_is_deterministic() -> None:
    index = InvertedIndex()

    index.add("doc-002:page:1", "retrieval")
    index.add("doc-001:page:2", "retrieval")
    index.add("doc-001:page:1", "retrieval")

    assert index.lookup("retrieval") == (
        "doc-001:page:1",
        "doc-001:page:2",
        "doc-002:page:1",
    )


def test_duplicate_page_indexing_does_not_duplicate_page_id() -> None:
    index = InvertedIndex()

    index.add("doc-001:page:1", "retrieval")
    index.add("doc-001:page:1", "retrieval")

    assert index.lookup("retrieval") == ("doc-001:page:1",)


def test_contains_reports_indexed_terms() -> None:
    index = InvertedIndex()

    index.add("doc-001:page:1", "retrieval system")

    assert index.contains("retrieval")
    assert not index.contains("unknown")


def test_remove_page_removes_all_its_terms() -> None:
    index = InvertedIndex()

    index.add(
        "doc-001:page:1",
        "retrieval system architecture",
    )

    index.remove("doc-001:page:1")

    assert index.lookup("retrieval") == ()
    assert index.lookup("system") == ()
    assert index.lookup("architecture") == ()


def test_remove_page_preserves_other_pages() -> None:
    index = InvertedIndex()

    index.add("doc-001:page:1", "retrieval system")
    index.add("doc-001:page:2", "retrieval architecture")

    index.remove("doc-001:page:1")

    assert index.lookup("retrieval") == (
        "doc-001:page:2",
    )
    assert index.lookup("system") == ()


def test_remove_unknown_page_is_safe() -> None:
    index = InvertedIndex()

    index.remove("doc-001:page:1")

    assert index.lookup("retrieval") == ()


def test_empty_text_indexes_no_terms() -> None:
    index = InvertedIndex()

    index.add("doc-001:page:1", "")

    assert index.lookup("retrieval") == ()


def test_clear_removes_all_entries() -> None:
    index = InvertedIndex()

    index.add("doc-001:page:1", "retrieval system")
    index.add("doc-001:page:2", "architecture")

    index.clear()

    assert index.lookup("retrieval") == ()
    assert index.lookup("system") == ()
    assert index.lookup("architecture") == ()


@pytest.mark.parametrize(
    "page_id",
    [
        "",
        " ",
    ],
)
def test_invalid_page_id_is_rejected(page_id: str) -> None:
    index = InvertedIndex()

    with pytest.raises(ValueError):
        index.add(page_id, "retrieval")
