"""Encode JSON payloads as server-sent events."""

from json import dumps


def event_frame(payload: dict[str, object], *, event: str | None = None) -> str:
    """Serialize a JSON object as an SSE frame.

    :param payload: The JSON data carried by the frame.
    :param event: Optional name for the event.
    :returns: A complete SSE frame.
    """
    name = f"event: {event}\n" if event else ""
    return f"{name}data: {dumps(payload, separators=(',', ':'))}\n\n"
