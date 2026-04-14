"""OpenAI Responses API helpers for segment-level semantic inference."""

from __future__ import annotations

import base64
import json
import mimetypes
import os
from pathlib import Path
from typing import Any

from src.semantic.mapping import normalize_semantic_controls
from src.semantic.pc_semantics import CANONICAL_SEMANTIC_ORDER, load_semantic_schema


DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"
RESPONSE_SCHEMA_NAME = "segment_semantic_controls"


def build_segment_prompt(
    template_path: str | Path,
    manifest: dict[str, Any],
    segment: dict[str, Any],
) -> str:
    """Render the prompt template for one segment."""
    template = Path(template_path).read_text(encoding="utf-8")
    payload = {
        "segment_id": segment["segment_id"],
        "start_s": segment["start_s"],
        "end_s": segment["end_s"],
        "duration_s": float(segment["end_s"] - segment["start_s"]),
        "event_name": segment.get("event_name", ""),
        "context": segment.get("context", ""),
        "notes": segment.get("notes", ""),
        "keyframe_paths": segment["keyframes"],
        "reference_video_path": manifest.get("reference_video_path"),
        "target_duration_seconds": manifest["target_duration_seconds"],
        "segment_count": len(manifest["segments"]),
    }
    return template.replace("{{segment_payload_json}}", json.dumps(payload, indent=2, ensure_ascii=False))


def parse_segment_response_text(
    text: str,
    *,
    schema: dict[str, Any] | None = None,
) -> tuple[dict[str, float], dict[str, str]]:
    """Parse and validate one structured OpenAI response."""
    payload = _extract_first_json_object(text)
    semantic_controls = payload.get("semantic_controls")
    rationale = payload.get("rationale", {})
    if not isinstance(semantic_controls, dict):
        raise ValueError("OpenAI response must contain a 'semantic_controls' object.")
    if not isinstance(rationale, dict):
        raise ValueError("OpenAI response rationale must be a JSON object.")

    normalized = normalize_semantic_controls(
        semantic_controls,
        schema=schema or load_semantic_schema(),
        clip=True,
        require_all=True,
    )
    normalized_rationale = {key: str(rationale.get(key, "") or "") for key in CANONICAL_SEMANTIC_ORDER}
    return normalized, normalized_rationale


class OpenAISemanticSegmentClient:
    """Thin wrapper around the OpenAI Responses API for segment inference."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_OPENAI_MODEL,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self.schema = load_semantic_schema()
        self.client = client or self._build_default_client()

    def infer_segment(
        self,
        *,
        prompt: str,
        keyframe_paths: list[str],
    ) -> dict[str, Any]:
        """Infer semantic controls for one segment."""
        content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
        for frame_path in keyframe_paths:
            content.append(
                {
                    "type": "input_image",
                    "image_url": _image_path_to_data_url(frame_path),
                }
            )

        response = self.client.responses.create(
            model=self.model,
            input=[{"role": "user", "content": content}],
            text={
                "format": {
                    "type": "json_schema",
                    "name": RESPONSE_SCHEMA_NAME,
                    "schema": _semantic_response_json_schema(),
                    "strict": True,
                }
            },
        )
        response_text = _get_response_text(response)
        semantic_controls, rationale = parse_segment_response_text(response_text, schema=self.schema)
        return {
            "response": response,
            "response_text": response_text,
            "semantic_controls": semantic_controls,
            "rationale": rationale,
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


def _semantic_response_json_schema() -> dict[str, Any]:
    semantic_properties = {
        key: {"type": "number", "minimum": 0.0, "maximum": 1.0} for key in CANONICAL_SEMANTIC_ORDER
    }
    rationale_properties = {key: {"type": "string"} for key in CANONICAL_SEMANTIC_ORDER}
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["semantic_controls", "rationale"],
        "properties": {
            "semantic_controls": {
                "type": "object",
                "additionalProperties": False,
                "required": CANONICAL_SEMANTIC_ORDER,
                "properties": semantic_properties,
            },
            "rationale": {
                "type": "object",
                "additionalProperties": False,
                "required": CANONICAL_SEMANTIC_ORDER,
                "properties": rationale_properties,
            },
        },
    }


def _image_path_to_data_url(path: str | Path) -> str:
    file_path = Path(path)
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
