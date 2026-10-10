"""Build the official Qwen3-VL chat message for one image."""

from __future__ import annotations

from PIL import Image

from src.errors import ImageValidationError


def build_image_message(image: Image.Image, prompt: str, *, image_source: Image.Image | str | None = None) -> list[dict]:
    """Return a one-turn user message whose image content is pixel data or a real file path.

    Only the ``image`` key is set. The transformers processor also accepts ``url``
    and ``path``; setting more than one of those keys loads the image twice.
    """
    if not isinstance(prompt, str) or not prompt.strip():
        raise ImageValidationError("The prompt text is empty.")
    source = image if image_source is None else image_source
    if not isinstance(source, (Image.Image, str)):
        raise ImageValidationError("The image passed to the model must be a PIL image or a file path.")
    if isinstance(source, Image.Image) and source.size[0] <= 0:
        raise ImageValidationError("The PIL image has no pixels.")
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": source},
                {"type": "text", "text": prompt},
            ],
        }
    ]
