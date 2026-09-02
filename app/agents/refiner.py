from app.agents.state import AgentState


class AgentQueryRefiner:
    """Generate deterministic follow-up queries for agent retrieval."""

    def refine(self, state: AgentState) -> str:
        """Generate the next retrieval query from the current agent state."""

        if not isinstance(state, AgentState):
            raise TypeError("state must be an AgentState")

        current_query = state.current_query.strip()

        if not current_query:
            raise ValueError("current query cannot be empty")

        if not state.contexts:
            return self._add_retrieval_instruction(current_query)

        latest_context = state.contexts[-1]

        if latest_context.page_count == 0:
            return self._add_retrieval_instruction(current_query)

        return current_query

    @staticmethod
    def _add_retrieval_instruction(query: str) -> str:
        """Make an unsuccessful query more explicit for lexical retrieval."""

        suffix = " relevant definition explanation details"

        if query.lower().endswith(suffix):
            return query

        return f"{query}{suffix}"