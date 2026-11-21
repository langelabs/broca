"""OpenAI-compatible Decisions route declaration."""

from fastapi import HTTPException, status

from ._request import CreateDecisionRequest
from ._response import DecisionResponse
from ._router import decision_router


@decision_router.post("")
async def create_decision(body: CreateDecisionRequest) -> DecisionResponse:
    """Declare evaluation of predicate, choice, or score questions.

    :param body: Model, shared evidence, and questions.
    :returns: Typed answers in question order, with token usage.
    :raises HTTPException: Decision inference is not implemented.
    """
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED)
