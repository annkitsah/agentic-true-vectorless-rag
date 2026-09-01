from app.agents.models import AgentActionType
from app.agents.state import AgentState
from app.retrieval.models import RetrievalQuery
from app.retrieval.service import RetrievalService


class AgentExecutor:
    """Execute an agent plan against application services."""

    def __init__(
        self,
        retrieval_service: RetrievalService,
    ) -> None:
        self.retrieval_service = retrieval_service

    def execute(
        self,
        state: AgentState,
    ) -> AgentState:
        """Execute the current plan and update the agent state."""

        if not isinstance(state, AgentState):
            raise TypeError("state must be an AgentState")

        if state.plan is None:
            raise ValueError("agent state must contain a plan")

        for action in state.plan.actions:
            self._execute_action(
                state,
                action.action_type,
                action.query,
            )

        state.advance_iteration()

        return state

    def _execute_action(
        self,
        state: AgentState,
        action_type: AgentActionType,
        query: str,
    ) -> None:
        if action_type is AgentActionType.RETRIEVE:
            context = self.retrieval_service.retrieve(
                RetrievalQuery(
                    text=query,
                )
            )

            state.add_context(context)
            return

        if action_type is AgentActionType.ANSWER:
            raise NotImplementedError(
                "answer action execution is not implemented yet"
            )

        if action_type is AgentActionType.REFINE:
            raise NotImplementedError(
                "refine action execution is not implemented yet"
            )

        raise ValueError(
            f"unsupported agent action: {action_type}"
        )