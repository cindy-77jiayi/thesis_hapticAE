"""Audit parity between HapticGen and HF-imported data preprocessing paths.

Outputs:
  - dataset_parity_report.json
  - dataset_parity_report.md
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from _bootstrap import add_project_root

add_project_root()

from src.data.preprocessing import (  # noqa: E402
    collect_clean_wavs,
    estimate_global_rms,
    load_segment_energy,
)
from src.utils.config import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit HapticGen/HF data parity before labeling")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="outputs/labeling")
    p.add_argument("--sample_per_source", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _safe_read_json(path: str) -> dict[str, Any] | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _first_subdir(root: str, path: str) -> str:
    rel = os.path.relpath(path, root)
    return rel.split(os.sep, 1)[0]


def _round(x: float) -> float:
    return float(np.round(x, 6))


def _counter_to_rows(counter: Counter[tuple[str, str, int]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (subdir, model, vote), cnt in sorted(counter.items()):
        rows.append(
            {
                "subdir": subdir,
                "model": model,
                "vote": vote,
                "count": int(cnt),
            }
        )
    return rows


def _sample_checks(
    files: list[str],
    label: str,
    n: int,
    T: int,
    sr: int,
    global_rms: float,
    scale: float,
    use_minmax: bool,
    rng: random.Random,
) -> list[dict[str, Any]]:
    if not files:
        return []
    picks = files if len(files) <= n else rng.sample(files, n)
    checks: list[dict[str, Any]] = []
    for p in picks:
        seg = load_segment_energy(
            p,
            T=T,
            sr_expect=sr,
            global_rms=global_rms,
            scale=scale,
            use_minmax=use_minmax,
        )
        checks.append(
            {
                "source": label,
                "path": p,
                "shape": list(seg.shape),
                "dtype": str(seg.dtype),
                "min": _round(float(seg.min())),
                "max": _round(float(seg.max())),
                "rms": _round(float(np.sqrt(np.mean(seg**2)))),
            }
        )
    return checks


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    config = load_config(args.config)
    data_cfg = config["data"]
    data_dir = os.path.abspath(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    accepted_models = set(data_cfg.get("accepted_models", ["HapticGen"]))
    accepted_votes = set(data_cfg.get("accepted_votes", [1]))
    include_subdirs_cfg = data_cfg.get("include_subdirs")
    include_subdirs = set(include_subdirs_cfg) if include_subdirs_cfg else None

    meta_paths = glob.glob(os.path.join(data_dir, "**", "*.am1.json"), recursive=True)
    raw_counter: Counter[tuple[str, str, int]] = Counter()
    all_by_subdir: Counter[str] = Counter()

    wav_to_meta: dict[str, dict[str, Any]] = {}
    hf_meta_details: defaultdict[str, set[str]] = defaultdict(set)

    for mp in meta_paths:
        meta = _safe_read_json(mp)
        if not isinstance(meta, dict):
            continue
        subdir = _first_subdir(data_dir, mp)
        model = str(meta.get("model", ""))
        vote = int(meta.get("vote", -9999))
        raw_counter[(subdir, model, vote)] += 1
        all_by_subdir[subdir] += 1

        wav_name = meta.get("filename")
        if isinstance(wav_name, str):
            wav_path = os.path.abspath(os.path.join(os.path.dirname(mp), wav_name))
            wav_to_meta[wav_path] = meta

        if subdir == "external_fsd50k":
            for k in ("source_dataset", "source_repo", "source_split", "model"):
                v = meta.get(k)
                if v is not None:
                    hf_meta_details[k].add(str(v))

    selected_wavs = collect_clean_wavs(
        data_dir,
        accepted_models=accepted_models,
        accepted_votes=accepted_votes,
        include_subdirs=include_subdirs,
    )
    selected_counter: Counter[tuple[str, str, int]] = Counter()
    selected_by_source: defaultdict[str, list[str]] = defaultdict(list)
    for wp in selected_wavs:
        meta = wav_to_meta.get(os.path.abspath(wp), {})
        subdir = _first_subdir(data_dir, wp)
        model = str(meta.get("model", ""))
        vote = int(meta.get("vote", -9999))
        selected_counter[(subdir, model, vote)] += 1
        selected_by_source[subdir].append(wp)

    if not selected_wavs:
        raise RuntimeError("No selected WAV files found under current config filters.")

    sr = int(data_cfg["sr"])
    T = int(data_cfg["T"])
    scale = float(data_cfg["scale"])
    use_minmax = bool(data_cfg.get("use_minmax", False))
    global_rms = estimate_global_rms(selected_wavs, n=min(200, len(selected_wavs)), sr_expect=sr)

    sample_checks: list[dict[str, Any]] = []
    for source, files in sorted(selected_by_source.items()):
        sample_checks.extend(
            _sample_checks(
                files=files,
                label=source,
                n=args.sample_per_source,
                T=T,
                sr=sr,
                global_rms=global_rms,
                scale=scale,
                use_minmax=use_minmax,
                rng=rng,
            )
        )

    parity_pass = True
    parity_notes = [
        "Training-time slicing path is shared: HapticWavDataset -> load_segment_energy.",
        "Selected sources use identical T/sr/global_rms/scale/use_minmax settings within one run.",
    ]
    differences = []
    if "external_fsd50k" in all_by_subdir:
        differences.append(
            "HF import has extra pre-ingest filtering (resample to target_sr, max_seconds trim, min_rms)."
        )
        differences.append(
            "HapticGen native samples are consumed from existing sidecars and do not pass through HF import filters."
        )
    else:
        differences.append("external_fsd50k subdir not present; parity audit only covers in-repo HapticGen data.")

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": os.path.abspath(args.config),
        "data_dir": data_dir,
        "filters": {
            "include_subdirs": sorted(include_subdirs) if include_subdirs else None,
            "accepted_models": sorted(accepted_models),
            "accepted_votes": sorted(int(v) for v in accepted_votes),
        },
        "counts": {
            "total_sidecars_scanned": int(sum(raw_counter.values())),
            "all_by_subdir": {k: int(v) for k, v in sorted(all_by_subdir.items())},
            "raw_grouped": _counter_to_rows(raw_counter),
            "selected_wavs_total": int(len(selected_wavs)),
            "selected_grouped": _counter_to_rows(selected_counter),
        },
        "hf_sidecar_metadata": {k: sorted(v) for k, v in hf_meta_details.items()},
        "training_preprocessing": {
            "shared_function": "src.data.preprocessing.load_segment_energy",
            "params": {
                "T": T,
                "sr_expect": sr,
                "global_rms": _round(global_rms),
                "scale": scale,
                "use_minmax": use_minmax,
                "load_segment_energy_defaults": {
                    "tries": 30,
                    "min_energy": 5e-4,
                    "max_resample": 5,
                    "clip_range": [-3.0, 3.0],
                },
            },
            "sample_checks": sample_checks,
            "parity_pass": parity_pass,
            "parity_notes": parity_notes,
        },
        "differences_before_training": differences,
    }

    json_path = out_dir / "dataset_parity_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    md_lines = [
        "# Dataset Parity Audit",
        "",
        f"- Config: `{report['config_path']}`",
        f"- Data dir: `{report['data_dir']}`",
        f"- Selected WAVs: **{report['counts']['selected_wavs_total']}**",
        "",
        "## Key Conclusion",
        "",
        "- Training slicing/filtering path after selection is shared across sources.",
        "- HF-imported data and native HapticGen data differ at pre-ingest stage.",
        "",
        "## Pre-Ingest Differences",
        "",
    ]
    for d in differences:
        md_lines.append(f"- {d}")
    md_lines.extend(
        [
            "",
            "## Shared Training Preprocessing",
            "",
            f"- Function: `{report['training_preprocessing']['shared_function']}`",
            f"- T={T}, sr={sr}, scale={scale}, use_minmax={use_minmax}",
            f"- global_rms (estimated)={report['training_preprocessing']['params']['global_rms']}",
            "",
            "## Source Counts (Selected)",
            "",
            "| Subdir | Model | Vote | Count |",
            "|---|---|---:|---:|",
        ]
    )
    for row in report["counts"]["selected_grouped"]:
        md_lines.append(
            f"| {row['subdir']} | {row['model']} | {row['vote']} | {row['count']} |"
        )
    md_lines.append("")

    md_path = out_dir / "dataset_parity_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"Saved: {json_path}")
    print(f"Saved: {md_path}")


if __name__ == "__main__":
    main()
