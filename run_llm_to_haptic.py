"""MVP wrapper: UI action/context -> canonical semantic controls -> PC vector -> haptic signal.

This script does not retrain VAE/PCA. It only performs semantic-to-control mapping.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

from baseline.rule_based_controls import get_rule_based_attributes
from src.semantic.mapping import normalize_semantic_controls, semantic_to_pca
from src.semantic.pc_semantics import CANONICAL_SEMANTIC_ORDER, load_semantic_schema


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _extract_first_json_object(text: str) -> dict[str, Any]:
    """Extract and parse the first JSON object from a potentially noisy string."""
    text = text.strip()
    if not text:
        raise ValueError("Empty LLM output")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[i:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    raise ValueError("No valid JSON object found in LLM output")


def _load_llm_output_text(args: argparse.Namespace) -> str | None:
    if args.llm_output_json:
        return args.llm_output_json
    if args.llm_output_path:
        return Path(args.llm_output_path).read_text(encoding="utf-8")
    return None


def _extract_semantic_controls(payload: dict[str, Any]) -> dict[str, Any]:
    """Extract canonical semantic controls from a parsed LLM payload."""
    source = payload.get("semantic_controls")
    if isinstance(source, dict):
        payload = source
    return {name: payload[name] for name in CANONICAL_SEMANTIC_ORDER if name in payload}


def _default_semantic_controls(schema: dict[str, Any]) -> dict[str, float]:
    """Return canonical semantic defaults from schema."""
    return {spec["canonical_name"]: float(spec.get("default", 0.5)) for spec in schema["semantic_controls"]}


def _build_prompt(
    template_path: Path,
    action_metadata: dict[str, Any],
    frame_paths: dict[str, str],
) -> str:
    template = template_path.read_text(encoding="utf-8")
    payload = {
        "frames": frame_paths,
        "action_name": action_metadata.get("action_name", ""),
        "context": action_metadata.get("context", ""),
        "notes": action_metadata.get("notes", ""),
    }
    prompt = template
    prompt = prompt.replace("{{before_image_path}}", frame_paths["before"])
    prompt = prompt.replace("{{during_image_path}}", frame_paths["during"])
    prompt = prompt.replace("{{after_image_path}}", frame_paths["after"])
    prompt = prompt.replace("{{action_name}}", str(payload["action_name"]))
    prompt = prompt.replace("{{context}}", str(payload["context"]))
    prompt = prompt.replace("{{notes}}", str(payload["notes"]))
    return prompt


def _maybe_decode_haptic(
    pc_vector: list[float],
    config_path: str | None,
    checkpoint: str | None,
    pca_dir: str | None,
    output_dir: Path,
) -> Path | None:
    """Decode 8D PC vector into waveform if config/checkpoint/pca are provided."""
    if not (config_path and checkpoint and pca_dir):
        return None
    # Defer heavy model imports so the MVP mapping pipeline can run without torch.
    import numpy as np
    import torch

    from src.data.loaders import build_model, load_checkpoint
    from src.pipelines.pca_control import control_to_latent
    from src.utils.config import load_config

    pca_path = Path(pca_dir) / "pca_pipe.pkl"
    if not pca_path.exists():
        raise FileNotFoundError(f"PCA pipeline not found: {pca_path}")

    with open(pca_path, "rb") as f:
        pipe = pickle.load(f)

    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config, device)
    load_checkpoint(model, checkpoint, device)

    z = control_to_latent(pipe, np.array(pc_vector, dtype=np.float32))
    z_t = torch.from_numpy(z).float().unsqueeze(0).to(device)
    signal = model.decode(z_t, target_len=config["data"]["T"]).squeeze().detach().cpu().numpy()

    out_path = output_dir / "generated_haptic.npy"
    np.save(out_path, signal.astype(np.float32))
    return out_path


def _maybe_save_preview(signal_path: Path, output_dir: Path) -> Path | None:
    """Save a simple waveform preview plot if matplotlib is available."""
    import numpy as np

    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    sig = np.load(signal_path)
    fig = plt.figure(figsize=(10, 3))
    ax = fig.add_subplot(111)
    ax.plot(sig, linewidth=0.8)
    ax.set_title("Generated Haptic Waveform Preview")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Amplitude")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    out = output_dir / "generated_haptic_preview.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run canonical semantic-to-haptic inference pipeline")
    parser.add_argument("--action_dir", type=str, required=True, help="Path to one action folder")
    parser.add_argument("--output_dir", type=str, default="outputs/llm_mvp")
    parser.add_argument("--prompt_template", type=str, default="llm/prompt_template.md")
    parser.add_argument("--llm_output_path", type=str, default=None, help="Path to raw LLM text/JSON output")
    parser.add_argument("--llm_output_json", type=str, default=None, help="Raw LLM output text passed inline")
    parser.add_argument("--use_rule_based", action="store_true", help="Use rule-based baseline instead of LLM")
    parser.add_argument("--baseline_action_type", type=str, default=None, help="Override action_type for baseline")
    parser.add_argument("--config", type=str, default=None, help="Config path for decoder hook")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path for decoder hook")
    parser.add_argument("--pca_dir", type=str, default=None, help="Directory containing pca_pipe.pkl")
    args = parser.parse_args()

    action_dir = Path(args.action_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    semantic_schema = load_semantic_schema()

    metadata_path = action_dir / "metadata.json"
    metadata = _read_json(metadata_path)
    frame_paths = {
        "before": str((action_dir / "before.png").resolve()),
        "during": str((action_dir / "during.png").resolve()),
        "after": str((action_dir / "after.png").resolve()),
    }

    prompt = _build_prompt(Path(args.prompt_template), metadata, frame_paths)
    (output_dir / "llm_prompt.txt").write_text(prompt, encoding="utf-8")

    source = "llm"
    fallback_reason = None
    rationale = {}

    if args.use_rule_based:
        source = "rule_based"
        semantic_controls = get_rule_based_attributes(args.baseline_action_type or metadata.get("action_type"), metadata)
    else:
        llm_text = _load_llm_output_text(args)
        if not llm_text:
            source = "fallback_default"
            fallback_reason = "No LLM output provided; used semantic fallback controls."
            semantic_controls = get_rule_based_attributes(args.baseline_action_type or metadata.get("action_type"), metadata)
        else:
            try:
                parsed = _extract_first_json_object(llm_text)
                rationale = parsed.get("rationale", {}) if isinstance(parsed.get("rationale", {}), dict) else {}
                semantic_controls = _extract_semantic_controls(parsed)
                if set(semantic_controls) != set(CANONICAL_SEMANTIC_ORDER):
                    missing = [name for name in CANONICAL_SEMANTIC_ORDER if name not in semantic_controls]
                    raise ValueError(f"Missing canonical semantic controls in LLM output: {missing}")
            except Exception as e:
                source = "fallback_rule_based"
                fallback_reason = f"Malformed or outdated semantic LLM output ({type(e).__name__}): {e}"
                semantic_controls = get_rule_based_attributes(args.baseline_action_type or metadata.get("action_type"), metadata)

    try:
        semantic_controls = normalize_semantic_controls(
            semantic_controls,
            schema=semantic_schema,
            clip=True,
            require_all=True,
        )
    except Exception as e:
        source = "fallback_default"
        fallback_reason = f"Semantic control validation failed ({type(e).__name__}): {e}"
        semantic_controls = _default_semantic_controls(semantic_schema)

    pc_vector = [float(v) for v in semantic_to_pca(semantic_controls, schema=semantic_schema).tolist()]

    _write_json(
        output_dir / "semantic_controls.json",
        {
            "source": source,
            "fallback_reason": fallback_reason,
            "action_name": metadata.get("action_name"),
            "action_type": metadata.get("action_type"),
            "semantic_controls": semantic_controls,
            "rationale": rationale,
            "schema_path": str(Path("semantic_control_schema.json")),
        },
    )
    _write_json(
        output_dir / "pca_control_vector.json",
        {
            "pc_vector": pc_vector,
            "semantic_controls": semantic_controls,
            "notes": {
                "frequency": "Mapped directly to PC1.",
                "intensity": "Mapped to inverted PC2 so higher semantic intensity produces lower PC2.",
                "envelope_modulation": "Mapped directly to PC3.",
                "temporal_grouping": "Mapped directly to PC4.",
                "sharpness": "Mapped directly to PC5.",
                "pc6_pc8": "Fixed to 0 until semantics are resolved.",
            },
        },
    )
    generated = _maybe_decode_haptic(pc_vector, args.config, args.checkpoint, args.pca_dir, output_dir)
    if generated:
        preview = _maybe_save_preview(generated, output_dir)
        if preview:
            print(f"Saved: {preview}")
        print(f"Saved: {generated}")

    print(f"Saved: {output_dir / 'semantic_controls.json'}")
    print(f"Saved: {output_dir / 'pca_control_vector.json'}")
    print("MVP pipeline finished.")


if __name__ == "__main__":
    main()
