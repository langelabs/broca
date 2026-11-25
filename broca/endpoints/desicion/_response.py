"""Response for the OpenAI-compatible Decisions endpoint."""

from typing import Annotated, Literal

from pydantic import BaseModel, Field, FiniteFloat, NonNegativeInt

from broca.inference.decision.types import Probability
from ._request import ChoiceValue


class DecisionAnswer(BaseModel):
    """Associate an answer with its question name when one was supplied."""

    name: str | None = None


class PredicateAnswer(DecisionAnswer):
    """Report the probability that the predicate holds."""

    type: Literal["predicate"]
    probability: Probability


class DecisionChoiceProbability(BaseModel):
    """Report the probability assigned to one choice value."""

    value: ChoiceValue
    probability: Probability


class ChoiceAnswer(DecisionAnswer):
    """Report the chosen value and the probability distribution."""

    type: Literal["choice"]
    choice: ChoiceValue
    confidence: Probability
    probabilities: list[DecisionChoiceProbability]


class DecisionScoreProbability(BaseModel):
    """Report the probability assigned to one rubric level."""

    label: str
    value: NonNegativeInt
    probability: Probability


class ScoreAnswer(DecisionAnswer):
    """Report the score and probabilities over the ordered rubric."""

    type: Literal["score"]
    score: FiniteFloat
    confidence: Probability
    probabilities: list[DecisionScoreProbability]


class RefusalAnswer(DecisionAnswer):
    """Indicate a refusal to answer an individual question."""

    type: Literal["refusal"]


class DecisionInputTokensDetails(BaseModel):
    """Report prompt cache reads and writes."""

    cached_tokens: NonNegativeInt
    cache_write_tokens: NonNegativeInt


class DecisionOutputTokensDetails(BaseModel):
    """Report reasoning tokens included in output usage."""

    reasoning_tokens: NonNegativeInt


class DecisionUsage(BaseModel):
    """Report token usage across the entire decision request."""

    input_tokens: NonNegativeInt
    input_tokens_details: DecisionInputTokensDetails
    output_tokens: NonNegativeInt
    output_tokens_details: DecisionOutputTokensDetails
    total_tokens: NonNegativeInt


class DecisionResponse(BaseModel):
    """Return typed answers in question order with usage when available."""

    model: str
    answers: list[
        Annotated[
            PredicateAnswer | ChoiceAnswer | ScoreAnswer | RefusalAnswer,
            Field(discriminator="type"),
        ]
    ]
    usage: DecisionUsage | None = None
