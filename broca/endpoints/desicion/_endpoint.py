"""Map decision questions to the configured Bekko inference export."""

from json import dumps

from fastapi import HTTPException
from pydantic import JsonValue

from ...inference import decision
from ...inference.decision.types import (
    ChoiceCandidate,
    ChoiceDecision,
    Decision,
    DecisionRequest,
    NoulDecision,
    ScoreCandidate,
    ScoreDecision,
)
from .._inference import require_loaded_model, run_inference
from ._request import CreateDecisionRequest, DecisionImage, DecisionToolOutput
from ._response import (
    ChoiceAnswer,
    DecisionChoiceProbability,
    DecisionResponse,
    DecisionScoreProbability,
    PredicateAnswer,
    ScoreAnswer,
)
from ._router import decision_router


@decision_router.post("", response_model_exclude_none=True)
async def create_decision(body: CreateDecisionRequest) -> DecisionResponse:
    """Evaluate predicate, choice, and score questions over shared evidence.

    :param body: Model, shared evidence, and ordered questions.
    :returns: The answers in question order.
    :raises HTTPException: If the input or configured model cannot be used.
    """
    require_loaded_model(decision._model, body.model)
    if body.prompt_cache_key is not None or body.prompt_cache_options is not None:
        raise HTTPException(422, "The decision model does not support prompt caching.")

    state: JsonValue
    if isinstance(body.input, str):
        state = body.input
    else:
        state = []
        for item in body.input:
            parts = (
                item.output
                if isinstance(item, DecisionToolOutput)
                else getattr(item, "content", None)
            )
            if isinstance(parts, list) and any(
                isinstance(part, DecisionImage)
                or getattr(part, "prompt_cache_breakpoint", None)
                for part in parts
            ):
                raise HTTPException(
                    422,
                    "The decision model supports text and JSON evidence without prompt caching.",
                )
            state.append(item.model_dump(mode="json", exclude_none=True))

    definitions: list[Decision] = []
    for question in body.questions:
        instructions = question.instructions
        if isinstance(instructions, list):
            if any(isinstance(part, DecisionImage) for part in instructions):
                raise HTTPException(
                    422, "The decision model supports text and JSON instructions."
                )
            instructions = "\n".join(
                part.text
                if part.type == "input_text"
                else dumps(part.data, ensure_ascii=False)
                for part in instructions
                if part.type != "input_image"
            )
        if not instructions.strip():
            raise HTTPException(422, "Decision instructions must not be blank.")

        definition: Decision
        match question.type:
            case "predicate":
                definition = NoulDecision(
                    instructions=instructions,
                    yes_description="The predicate is true.",
                    no_description="The predicate is false.",
                )
            case "choice":
                if len({choice.value for choice in question.choices}) != len(
                    question.choices
                ):
                    raise HTTPException(422, "Decision choice values must be unique.")
                definition = ChoiceDecision(
                    instructions=instructions,
                    candidates=[
                        ChoiceCandidate(
                            id=str(index),
                            description=(
                                f"{dumps(choice.value, ensure_ascii=False)}: {choice.description}"
                                if choice.description
                                else dumps(choice.value, ensure_ascii=False)
                            ),
                        )
                        for index, choice in enumerate(question.choices)
                    ],
                )
            case "score":
                definition = ScoreDecision(
                    instructions=instructions,
                    candidates=[
                        ScoreCandidate(
                            id=str(index),
                            value=index,
                            description=f"{level.label}: {level.description}"
                            if level.description
                            else level.label or str(index),
                        )
                        for index, level in enumerate(question.levels)
                    ],
                )
        definitions.append(definition)

    answers: list[PredicateAnswer | ChoiceAnswer | ScoreAnswer] = []
    for question, definition in zip(body.questions, definitions, strict=True):
        result = await run_inference(
            decision._model,
            decision.infer(DecisionRequest(state=state, decision=definition)),
        )
        if question.type == "predicate" and result.kind == "noul":
            answers.append(
                PredicateAnswer(
                    type="predicate",
                    name=question.name,
                    probability=result.probability_yes,
                )
            )
        elif question.type == "choice" and result.kind == "choice":
            probabilities = [
                DecisionChoiceProbability(
                    value=choice.value, probability=result.probabilities[str(index)]
                )
                for index, choice in enumerate(question.choices)
            ]
            selected = next(
                (
                    item
                    for index, item in enumerate(probabilities)
                    if str(index) == result.selected_id
                ),
                None,
            )
            if selected is None:
                raise HTTPException(
                    503, "The model returned an unknown decision choice."
                )
            answers.append(
                ChoiceAnswer(
                    type="choice",
                    name=question.name,
                    choice=selected.value,
                    confidence=selected.probability,
                    probabilities=probabilities,
                )
            )
        elif question.type == "score" and result.kind == "score":
            probabilities_by_level = [
                DecisionScoreProbability(
                    label=level.label,
                    value=index,
                    probability=result.probabilities[str(index)],
                )
                for index, level in enumerate(question.levels)
            ]
            answers.append(
                ScoreAnswer(
                    type="score",
                    name=question.name,
                    score=result.score,
                    confidence=max(result.probabilities.values()),
                    probabilities=probabilities_by_level,
                )
            )
        else:
            raise HTTPException(
                503, "The model returned an unexpected decision result."
            )
    return DecisionResponse(model=body.model, answers=answers)
