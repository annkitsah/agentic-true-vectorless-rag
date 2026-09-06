import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import get_container
from app.api.schemas import QueryRequest, QueryResponse
from app.container import ApplicationContainer

logger = logging.getLogger(__name__)

router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
async def run_query(
    payload: QueryRequest,
    container: ApplicationContainer = Depends(get_container),
) -> QueryResponse:
    """Run the agentic retrieval-and-answer loop for a question."""

    try:
        response = container.agent_orchestrator.run(payload.question)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Agent orchestration failed for query.")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=(
                "The generation provider failed to produce an answer. "
                "This is usually a rate limit or upstream outage; "
                "please retry shortly."
            ),
        ) from exc

    return QueryResponse(
        query=response.query,
        answer=response.answer,
        iterations=response.iterations,
    )