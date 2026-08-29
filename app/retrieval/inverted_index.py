from collections import defaultdict

from app.retrieval.text import tokenize


class InvertedIndex:
    """Deterministic in-memory inverted index for page-level retrieval."""

    def __init__(self) -> None:
        self._index: dict[str, set[str]] = defaultdict(set)
        self._page_terms: dict[str, set[str]] = {}

    def add(
        self,
        page_id: str,
        text: str,
    ) -> None:
        """Index all normalized terms occurring in a page."""

        if not page_id.strip():
            raise ValueError("page_id cannot be empty")

        # Remove any previous term associations so re-indexing a page
        # always reflects its current content.
        self.remove(page_id)

        terms = set(tokenize(text))

        if not terms:
            return

        self._page_terms[page_id] = terms

        for term in terms:
            self._index[term].add(page_id)

    def lookup(
        self,
        term: str,
    ) -> tuple[str, ...]:
        """Return page IDs containing the normalized term."""

        normalized_terms = tokenize(term)

        if not normalized_terms:
            return ()

        # lookup() is intentionally term-oriented. For normal input,
        # tokenize() produces one term; use the first normalized term.
        normalized_term = normalized_terms[0]

        page_ids = self._index.get(normalized_term)

        if not page_ids:
            return ()

        return tuple(sorted(page_ids))

    def contains(
        self,
        term: str,
    ) -> bool:
        """Return whether the normalized term exists in the index."""

        return bool(self.lookup(term))

    def remove(
        self,
        page_id: str,
    ) -> None:
        """Remove a page and all of its term associations."""

        terms = self._page_terms.pop(page_id, None)

        if terms is None:
            return

        for term in terms:
            page_ids = self._index.get(term)

            if page_ids is None:
                continue

            page_ids.discard(page_id)

            if not page_ids:
                del self._index[term]

    def clear(self) -> None:
        """Remove every indexed page and term."""

        self._index.clear()
        self._page_terms.clear()