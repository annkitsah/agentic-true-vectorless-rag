from abc import ABC, abstractmethod

from app.retrieval.models import RetrievedContext


class Answerer(ABC):
    """Abstraction for generating an answer from retrieved context."""

    @abstractmethod
    def answer(
        self,
        query: str,
        context: RetrievedContext,
    ) -> str:
        """Generate an answer for a query using retrieved context."""

        raise NotImplementedError


class ContextAnswerer(Answerer):
    """Return the retrieved context directly as the answer."""

    def answer(
        self,
        query: str,
        context: RetrievedContext,
    ) -> str:
        """Return retrieved context text without additional generation."""

        if not isinstance(query, str):
            raise TypeError("query must be a string")

        if not query.strip():
            raise ValueError("query cannot be empty")

        if not isinstance(context, RetrievedContext):
            raise TypeError(
                "context must be a RetrievedContext"
            )

        if not context.text.strip():
            raise ValueError(
                "cannot build answer from empty retrieved context"
            )

        return context.text