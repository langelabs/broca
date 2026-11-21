"""Binary response body for the speech endpoint."""

from pydantic import RootModel


class SpeechResponse(RootModel[bytes]):
    """Describe encoded speech bytes sent directly as the HTTP body."""
