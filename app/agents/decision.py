from app.agents.models import AgentDecision, AgentDecisionType
from app.agents.state import AgentState


class AgentDecisionEngine:
    """Determine the next action from the current agent state."""

    def __init__(
        self,
        *,
        max_iterations: int = 3,
    ) -> None:
        if max_iterations < 1:
            raise ValueError(
                "max_iterations must be greater than zero"
            )

        self.max_iterations = max_iterations

    def decide(
        self,
        state: AgentState,
    ) -> AgentDecision:
        """Determine whether the agent should answer, refine, or stop."""

        if not isinstance(state, AgentState):
            raise TypeError(
                "state must be an AgentState"
            )

        if state.iteration >= self.max_iterations:
            return AgentDecision(
                decision_type=AgentDecisionType.STOP,
                reason=(
                    "Maximum agent iterations reached."
                ),
            )

        if not state.contexts:
            return AgentDecision(
                decision_type=AgentDecisionType.REFINE,
                reason=(
                    "No retrieved context is available."
                ),
                next_query=state.current_query,
            )

        latest_context = state.contexts[-1]

        if latest_context.page_count == 0:
            return AgentDecision(
                decision_type=AgentDecisionType.REFINE,
                reason=(
                    "The latest retrieval produced no relevant pages."
                ),
                next_query=state.current_query,
            )

        return AgentDecision(
            decision_type=AgentDecisionType.ANSWER,
            reason=(
                "Relevant retrieved context is available."
            ),
        )