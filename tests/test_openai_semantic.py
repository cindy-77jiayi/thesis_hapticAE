from __future__ import annotations

import pytest

from src.llm.openai_semantic import OpenAISemanticSegmentClient, parse_segment_response_text


def test_openai_client_requires_api_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(EnvironmentError, match="OPENAI_API_KEY"):
        OpenAISemanticSegmentClient()


def test_parse_segment_response_text_requires_canonical_structure():
    semantic_controls, rationale = parse_segment_response_text(
        """
        {
          "semantic_controls": {
            "frequency": 0.1,
            "intensity": 0.2,
            "envelope_modulation": 0.3,
            "temporal_grouping": 0.4,
            "sharpness": 0.5
          },
          "rationale": {
            "frequency": "a",
            "intensity": "b",
            "envelope_modulation": "c",
            "temporal_grouping": "d",
            "sharpness": "e"
          }
        }
        """
    )

    assert semantic_controls["frequency"] == 0.1
    assert semantic_controls["sharpness"] == 0.5
    assert rationale["temporal_grouping"] == "d"
