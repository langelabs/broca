"""Pydantic contracts for Choice, Noul, and Score decisions."""

from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, JsonValue, PositiveInt

Probability: TypeAlias = Annotated[FiniteFloat, Field(ge=0, le=1)]


class ChoiceCandidate(BaseModel):
    """Identify and describe one alternative for a Choice decision."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    description: str = Field(min_length=1)


class ScoreCandidate(ChoiceCandidate):
    """Attach an explicit numeric rubric value to a candidate."""

    value: FiniteFloat


class ChoiceDecision(BaseModel):
    """Select among named candidates, each with a unique identifier."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["choice"] = "choice"
    instructions: str = Field(min_length=1)
    candidates: list[ChoiceCandidate] = Field(min_length=2)
    system_prompt: str | None = None


class NoulDecision(BaseModel):
    """Evaluate an authored binary condition with explicit yes and no meanings."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["noul"] = "noul"
    instructions: str = Field(min_length=1)
    yes_description: str = Field(min_length=1)
    no_description: str = Field(min_length=1)
    system_prompt: str | None = None


class ScoreDecision(BaseModel):
    """Score a rubric with unique candidate IDs and distinct numeric values."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["score"] = "score"
    instructions: str = Field(min_length=1)
    candidates: list[ScoreCandidate] = Field(min_length=2)
    system_prompt: str | None = None


Decision: TypeAlias = Annotated[
    ChoiceDecision | NoulDecision | ScoreDecision, Field(discriminator="kind")
]


class DecisionRequest(BaseModel):
    """Pair JSON application state with one typed decision definition."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    state: JsonValue
    decision: Decision


class DecisionOptions(BaseModel):
    """Override the configured decision token limits and microbatch budget."""

    model_config = ConfigDict(extra="forbid")

    token_budget: PositiveInt | None = None
    context_length: PositiveInt | None = None
    query_length: PositiveInt | None = None
    document_length: PositiveInt | None = None


class ChoiceResult(BaseModel):
    """Return the selected candidate and probabilities keyed by candidate ID."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["choice"] = "choice"
    selected_id: str
    probabilities: dict[str, Probability] = Field(min_length=1)


class NoulResult(BaseModel):
    """Return the probability of the authored yes condition."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["noul"] = "noul"
    probability_yes: Probability


class ScoreResult(BaseModel):
    """Return the expected rubric score and candidate probabilities."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["score"] = "score"
    score: FiniteFloat
    normalized_score: Probability
    probabilities: dict[str, Probability] = Field(min_length=1)


DecisionResult: TypeAlias = Annotated[
    ChoiceResult | NoulResult | ScoreResult, Field(discriminator="kind")
]
