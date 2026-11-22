"""Run typed decisions using the configured Bekko model."""

import json

from pydantic import JsonValue, TypeAdapter

from ...ai_model import AiModel
from .._pipelines import BekkoPipeline
from ._model import _model
from ._types import DecisionOptions, DecisionRequest, DecisionResult

result_adapter: TypeAdapter[DecisionResult] = TypeAdapter(DecisionResult)


class DecisionPipeline(BekkoPipeline[DecisionRequest, DecisionResult]):
    """Evaluate typed decisions using the configured Bekko runtime."""

    def __init__(self, model: AiModel = _model) -> None:
        """Initialize the pipeline with a compatible model configuration.

        :param model: Model to use. Defaults to the built-in configuration.
        """
        super().__init__(model)

    async def _infer(
        self,
        request: DecisionRequest,
        *,
        options: DecisionOptions | None = None,
    ) -> DecisionResult:
        """Evaluate one Choice, Noul, or Score decision over supplied state.

        :param request: JSON state and the decision instructions and candidates.
        :param options: Optional overrides of configured token limits and budget.
        :returns: A result whose kind matches the requested decision kind.
        :raises ValueError: If candidate IDs or score values are duplicated.
        """
        decision = request.decision
        criteria: list[JsonValue]
        if decision.kind == "noul":
            criteria = [
                {"id": "yes", "description_json": json.dumps(decision.yes_description)},
                {"id": "no", "description_json": json.dumps(decision.no_description)},
            ]
        else:
            if len({candidate.id for candidate in decision.candidates}) != len(
                decision.candidates
            ):
                raise ValueError("Decision candidate IDs must be unique.")
            if decision.kind == "score" and len(
                {candidate.value for candidate in decision.candidates}
            ) != len(decision.candidates):
                raise ValueError("Score candidate values must be distinct.")
            criteria = [
                {
                    "id": candidate.id,
                    "description_json": json.dumps(candidate.description),
                }
                | candidate.model_dump(include={"value"}, mode="json")
                for candidate in decision.candidates
            ]

        config = (self.model.generation_options or {}).copy()
        if options is not None:
            config.update(options.model_dump(exclude_none=True, mode="json"))
        result = self.runtime.predict(
            {
                "state_json": json.dumps(request.state, allow_nan=False),
                "decisions": [
                    {
                        "id": "decision",
                        "kind": "judgment",
                        "type": decision.kind,
                        "instructions_json": json.dumps(decision.instructions),
                        "system_prompt": decision.system_prompt,
                        "criteria": criteria,
                    }
                ],
            },
            **config,
        )["decision"]
        if decision.kind == "noul":
            result.pop("probabilities", None)
        if decision.kind == "score":
            result.pop("values", None)
        return result_adapter.validate_python({"kind": decision.kind, **result})


pipeline = DecisionPipeline()
