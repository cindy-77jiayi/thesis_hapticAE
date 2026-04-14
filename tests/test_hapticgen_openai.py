from __future__ import annotations

import pytest

from src.llm_hapticgen.openai_hapticgen import (
    OpenAIHapticGenPromptClient,
    parse_hapticgen_response_text,
)


def test_hapticgen_openai_client_requires_api_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(EnvironmentError, match="OPENAI_API_KEY"):
        OpenAIHapticGenPromptClient()


def test_parse_hapticgen_response_text_requires_non_empty_prompt():
    payload = parse_hapticgen_response_text(
        """
        {
          "visual_summary": "A small object lands and settles.",
          "haptic_prompt": "A short sharp tap followed by a soft fading buzz.",
          "negative_prompt": "",
          "rationale": "The sequence suggests a brief impact and a decaying afterfeel."
        }
        """
    )
    assert payload["visual_summary"].startswith("A small object")
    assert "tap" in payload["haptic_prompt"]


def test_parse_hapticgen_response_text_rejects_missing_fields():
    with pytest.raises(ValueError, match="Missing required OpenAI fields"):
        parse_hapticgen_response_text('{"visual_summary": "x"}')
