"""Part 1 CLI: segment manifest -> OpenAI semantics -> PC vectors -> reconstruction payload."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.llm import (
    DEFAULT_OPENAI_MODEL,
    OpenAISemanticSegmentClient,
    build_reconstruction_payload,
    build_segment_prompt,
    load_segment_manifest,
)
from src.semantic.mapping import semantic_to_pca
from src.semantic.pc_semantics import CANONICAL_SEMANTIC_ORDER, load_semantic_schema


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _serialize_response(response: Any) -> dict[str, Any]:
    if hasattr(response, "model_dump"):
        return response.model_dump(mode="json")
    if hasattr(response, "to_dict"):
        return response.to_dict()
    return {"raw_response": str(response)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Part 1 LLM semantic inference from a segment manifest")
    parser.add_argument("--manifest", type=str, required=True, help="Path to one segment manifest JSON file")
    parser.add_argument("--output_dir", type=str, default="outputs/llm_part1")
    parser.add_argument("--prompt_template", type=str, default="llm/prompt_template.md")
    parser.add_argument("--model", type=str, default=DEFAULT_OPENAI_MODEL)
    parser.add_argument("--run_name", type=str, default="vae_balanced")
    parser.add_argument("--target_sr", type=int, default=8000)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_segment_manifest(args.manifest)
    semantic_schema = load_semantic_schema()
    client = OpenAISemanticSegmentClient(model=args.model)

    _write_json(output_dir / "normalized_segment_manifest.json", manifest)

    segment_outputs: list[dict[str, Any]] = []
    for segment in manifest["segments"]:
        segment_dir = output_dir / "segments" / segment["segment_id"]
        segment_dir.mkdir(parents=True, exist_ok=True)

        prompt = build_segment_prompt(args.prompt_template, manifest, segment)
        (segment_dir / "llm_prompt.txt").write_text(prompt, encoding="utf-8")

        request_metadata = {
            "model": args.model,
            "segment_id": segment["segment_id"],
            "start_s": segment["start_s"],
            "end_s": segment["end_s"],
            "keyframes": segment["keyframes"],
            "prompt_template": str(Path(args.prompt_template).resolve()),
            "manifest_path": manifest.get("manifest_path"),
        }
        _write_json(segment_dir / "request_metadata.json", request_metadata)

        result = client.infer_segment(prompt=prompt, keyframe_paths=segment["keyframes"])
        semantic_controls = result["semantic_controls"]
        rationale = result["rationale"]
        pc_vector = semantic_to_pca(semantic_controls, schema=semantic_schema).tolist()

        _write_json(segment_dir / "raw_response.json", _serialize_response(result["response"]))
        _write_json(
            segment_dir / "semantic_controls.json",
            {
                "segment_id": segment["segment_id"],
                "semantic_controls": {
                    key: float(semantic_controls[key]) for key in CANONICAL_SEMANTIC_ORDER
                },
                "rationale": rationale,
            },
        )
        _write_json(
            segment_dir / "pc_vector.json",
            {
                "segment_id": segment["segment_id"],
                "pc_vector": [float(v) for v in pc_vector],
                "semantic_controls": {
                    key: float(semantic_controls[key]) for key in CANONICAL_SEMANTIC_ORDER
                },
            },
        )

        segment_outputs.append(
            {
                "segment_id": segment["segment_id"],
                "semantic_controls": semantic_controls,
                "pc_vector": pc_vector,
                "rationale": rationale,
            }
        )

    payload = build_reconstruction_payload(
        segment_outputs,
        run_name=args.run_name,
        target_sr=args.target_sr,
    )
    _write_json(output_dir / "reconstruction_payload.json", payload)

    print(f"Saved: {output_dir / 'normalized_segment_manifest.json'}")
    print(f"Saved: {output_dir / 'reconstruction_payload.json'}")
    print("Part 1 pipeline finished.")


if __name__ == "__main__":
    main()
