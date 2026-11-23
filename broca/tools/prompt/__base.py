from pydantic import BaseModel


class Prompt(BaseModel):
    """
    Base model for all prompts. Prompting parameters are defined as pydantic fields.
    Calling the prompt fills the parameter in the prompt and returns the full prompt.
    """

    frame: str

    def __call__(self, **kwargs) -> str:
        # guard that all parameter that are required are present

        return str(self.frame.format(self.model_dump()))
