from __future__ import annotations

from src.llm.manifest import build_reconstruction_payload


def test_build_reconstruction_payload_preserves_segment_order():
    payload = build_reconstruction_payload(
        [
            {
                "segment_id": "seg_01",
                "semantic_controls": {
                    "frequency": 0.1,
                    "intensity": 0.2,
                    "envelope_modulation": 0.3,
                    "temporal_grouping": 0.4,
                    "sharpness": 0.5,
                },
                "pc_vector": [1, 2, 3, 4, 5, 0, 0, 0],
                "rationale": {"frequency": "a", "intensity": "b", "envelope_modulation": "c", "temporal_grouping": "d", "sharpness": "e"},
            },
            {
                "segment_id": "seg_02",
                "semantic_controls": {
                    "frequency": 0.6,
                    "intensity": 0.7,
                    "envelope_modulation": 0.8,
                    "temporal_grouping": 0.9,
                    "sharpness": 1.0,
                },
                "pc_vector": [6, 7, 8, 9, 10, 0, 0, 0],
                "rationale": {"frequency": "f", "intensity": "g", "envelope_modulation": "h", "temporal_grouping": "i", "sharpness": "j"},
            },
        ],
        run_name="vae_balanced",
        target_sr=8000,
    )

    assert payload["run_name"] == "vae_balanced"
    assert payload["target_sr"] == 8000
    assert payload["segment_duration_s"] == 0.5
    assert [segment["segment_id"] for segment in payload["segments"]] == ["seg_01", "seg_02"]
    assert payload["segments"][0]["pc_vector"][5:] == [0.0, 0.0, 0.0]
