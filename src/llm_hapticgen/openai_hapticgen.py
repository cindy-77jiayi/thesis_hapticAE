"""OpenAI bridge for converting three images into a HapticGen text prompt."""

from __future__ import annotations

import base64
import json
import mimetypes
import os
from pathlib import Path
from typing import Any


DEFAULT_HAPTICGEN_OPENAI_MODEL = "gpt-4.1-mini"
RESPONSE_SCHEMA_NAME = "hapticgen_prompt_response"


def parse_hapticgen_response_text(text: str) -> dict[str, str]:
    """Parse the structured OpenAI response for the HapticGen validation flow."""
    payload = _extract_first_json_object(text)
    required = ["visual_summary", "haptic_prompt", "negative_prompt", "rationale"]
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"Missing required OpenAI fields: {missing}")

    normalized = {key: str(payload[key]).strip() for key in required}
    if not normalized["haptic_prompt"]:
        raise ValueError("OpenAI response must include a non-empty haptic_prompt.")
    return normalized


class OpenAIHapticGenPromptClient:
    """Thin wrapper around the OpenAI Responses API for 3-image HapticGen prompting."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_HAPTICGEN_OPENAI_MODEL,
        template_path: str | Path = "llm/hapticgen_prompt_template.md",
        client: Any | None = None,
    ) -> None:
        self.model = model
        self.template_path = Path(template_path)
        self.client = client or self._build_default_client()

    def build_prompt(self, *, notes: str = "") -> str:
        """Render the system/user prompt template for this 3-image flow."""
        template = self.template_path.read_text(encoding="utf-8")
        return template.replace("{{notes}}", notes.strip())

    def infer_prompt(self, *, image_paths: list[str], notes: str = "") -> dict[str, Any]:
        """Send three local images to OpenAI and return a validated prompt payload."""
        if len(image_paths) != 3:
            raise ValueError(f"Expected exactly 3 images, got {len(image_paths)}.")

        content: list[dict[str, Any]] = [
            {"type": "input_text", "text": self.build_prompt(notes=notes)},
        ]
        for image_path in image_paths:
            content.append(
                {
                    "type": "input_image",
                    "image_url": _image_path_to_data_url(image_path),
                }
            )

        response = self.client.responses.create(
            model=self.model,
            input=[{"role": "user", "content": content}],
            text={
                "format": {
                    "type": "json_schema",
                    "name": RESPONSE_SCHEMA_NAME,
                    "schema": _hapticgen_response_json_schema(),
                    "strict": True,
                }
            },
        )
        response_text = _get_response_text(response)
        parsed = parse_hapticgen_response_text(response_text)
        return {
            "response": response,
            "response_text": response_text,
            **parsed,
        }

    @staticmethod
    def _build_default_client() -> Any:
        if not os.getenv("OPENAI_API_KEY"):
            raise EnvironmentError(
                "OPENAI_API_KEY is not set. In PowerShell, run "
                "`$env:OPENAI_API_KEY=\"<your-key>\"` for the current shell or "
                "`setx OPENAI_API_KEY \"<your-key>\"` for future shells."
            )
        from openai import OpenAI

        return OpenAI()


def _hapticgen_response_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["visual_summary", "haptic_prompt", "negative_prompt", "rationale"],
        "properties": {
            "visual_summary": {"type": "string"},
            "haptic_prompt": {"type": "string"},
            "negative_prompt": {"type": "string"},
            "rationale": {"type": "string"},
        },
    }


def _image_path_to_data_url(path: str | Path) -> str:
    file_path = Path(path).resolve()
    mime_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(file_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _get_response_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    if hasattr(response, "model_dump_json"):
        raw = response.model_dump_json(indent=2)
    elif hasattr(response, "model_dump"):
        raw = json.dumps(response.model_dump(mode="json"), indent=2, ensure_ascii=False)
    else:
        raw = str(response)
    raise ValueError(f"OpenAI response did not include output_text. Raw response:\n{raw}")


def _extract_first_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if not text:
        raise ValueError("Empty OpenAI output.")
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for idx, char in enumerate(text):
        if char != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError("No valid JSON object found in OpenAI output.")
