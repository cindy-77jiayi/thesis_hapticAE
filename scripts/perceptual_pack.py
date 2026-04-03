"""Generate and summarize perceptual AB trials, then freeze labeling artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from _bootstrap import add_project_root

add_project_root()

@dataclass
class Trial:
    trial_id: str
    pc: str
    axis: int
    label: str
    adjective_low: str
    adjective_high: str
    file_A: str
    file_B: str
    high_side: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Perceptual trial pack and labeling freeze")
    sub = p.add_subparsers(dest="mode", required=True)

    g = sub.add_parser("generate", help="Generate AB trials and trial sheet CSV")
    g.add_argument("--config", type=str, required=True)
    g.add_argument("--checkpoint", type=str, required=True)
    g.add_argument("--pca_dir", type=str, required=True)
    g.add_argument("--controls_spec", type=str, required=True)
    g.add_argument("--output_dir", type=str, default="outputs/labeling")
    g.add_argument("--n_trials_per_pc", type=int, default=12)
    g.add_argument("--seed", type=int, default=42)

    s = sub.add_parser("summarize", help="Summarize completed trial sheet")
    s.add_argument("--trial_csv", type=str, required=True)
    s.add_argument("--output_dir", type=str, default="outputs/labeling")
    s.add_argument("--min_trials_per_pc", type=int, default=12)

    f = sub.add_parser("freeze", help="Freeze controls spec/table from objective+perceptual results")
    f.add_argument("--controls_spec", type=str, required=True)
    f.add_argument("--controls_table", type=str, required=True)
    f.add_argument("--objective_gate", type=str, required=True)
    f.add_argument("--perceptual_summary", type=str, required=True)
    f.add_argument("--output_dir", type=str, default="outputs/labeling")

    return p.parse_args()


def _load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: str | Path, payload: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _control_to_latent(pipe: Any, c: np.ndarray) -> np.ndarray:
    squeeze = False
    if c.ndim == 1:
        c = c.reshape(1, -1)
        squeeze = True
    z = pipe.inverse_transform(c)
    if squeeze:
        z = z.squeeze(0)
    return z


def _decode_signal(
    pipe: Any,
    model: Any,
    device: Any,
    control: np.ndarray,
    T: int,
) -> np.ndarray:
    import torch

    z = _control_to_latent(pipe, control.astype(np.float32))
    z_t = torch.from_numpy(z).float().unsqueeze(0).to(device)
    with torch.no_grad():
        sig = model.decode(z_t, target_len=T).squeeze().detach().cpu().numpy().astype(np.float32)
    return sig


def _norm_audio(x: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(x))) + 1e-8
    y = x / peak
    return np.clip(y, -1.0, 1.0).astype(np.float32)


def generate(args: argparse.Namespace) -> None:
    import torch

    from src.data.loaders import build_model, load_checkpoint
    from src.utils.config import load_config

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.output_dir)
    sig_dir = out_dir / "perceptual_signals"
    sig_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    controls = _load_json(args.controls_spec)
    with open(Path(args.pca_dir) / "pca_pipe.pkl", "rb") as f:
        pipe = pickle.load(f)

    sr = int(config["data"]["sr"])
    T = int(config["data"]["T"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config, device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()

    n_controls = int(controls["n_controls"])
    base = np.zeros(n_controls, dtype=np.float32)

    trial_rows: list[Trial] = []
    for ctrl in controls["controls"]:
        axis = int(ctrl["axis"])
        pc = f"PC{axis + 1}"
        r = ctrl["range"]
        low_val = float(r["low"])
        high_val = float(r["high"])

        label = ctrl.get("composite_label", ctrl["name"])
        adjs = ctrl.get("adjectives", ["low", "high"])
        adj_low = adjs[0] if len(adjs) > 0 else "low"
        adj_high = adjs[1] if len(adjs) > 1 else "high"

        c_low = base.copy()
        c_high = base.copy()
        c_low[axis] = low_val
        c_high[axis] = high_val

        sig_low = _decode_signal(pipe, model, device, c_low, T)
        sig_high = _decode_signal(pipe, model, device, c_high, T)

        low_wav = sig_dir / f"{pc}_low.wav"
        high_wav = sig_dir / f"{pc}_high.wav"
        sf.write(str(low_wav), _norm_audio(sig_low), samplerate=sr, subtype="PCM_16")
        sf.write(str(high_wav), _norm_audio(sig_high), samplerate=sr, subtype="PCM_16")

        np.save(sig_dir / f"{pc}_low.npy", sig_low.astype(np.float32))
        np.save(sig_dir / f"{pc}_high.npy", sig_high.astype(np.float32))

        for t in range(args.n_trials_per_pc):
            high_side = "A" if rng.random() < 0.5 else "B"
            file_A = str(high_wav if high_side == "A" else low_wav)
            file_B = str(high_wav if high_side == "B" else low_wav)
            trial_rows.append(
                Trial(
                    trial_id=f"{pc}_T{t + 1:02d}",
                    pc=pc,
                    axis=axis,
                    label=label,
                    adjective_low=adj_low,
                    adjective_high=adj_high,
                    file_A=file_A,
                    file_B=file_B,
                    high_side=high_side,
                )
            )

    rng.shuffle(trial_rows)
    trial_csv = out_dir / "perceptual_trial_sheet.csv"
    with open(trial_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "trial_id",
                "pc",
                "axis",
                "label",
                "adjective_low",
                "adjective_high",
                "file_A",
                "file_B",
                "high_side",
                "user_diff_detected",
                "user_choice_high_end",
                "user_confidence_1to5",
                "user_notes",
            ],
        )
        w.writeheader()
        for tr in trial_rows:
            w.writerow(
                {
                    "trial_id": tr.trial_id,
                    "pc": tr.pc,
                    "axis": tr.axis,
                    "label": tr.label,
                    "adjective_low": tr.adjective_low,
                    "adjective_high": tr.adjective_high,
                    "file_A": tr.file_A,
                    "file_B": tr.file_B,
                    "high_side": tr.high_side,
                    "user_diff_detected": "",
                    "user_choice_high_end": "",
                    "user_confidence_1to5": "",
                    "user_notes": "",
                }
            )

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": os.path.abspath(args.config),
        "checkpoint": os.path.abspath(args.checkpoint),
        "controls_spec": os.path.abspath(args.controls_spec),
        "pca_dir": os.path.abspath(args.pca_dir),
        "n_trials_per_pc": args.n_trials_per_pc,
        "n_total_trials": len(trial_rows),
        "trial_csv": str(trial_csv.resolve()),
        "instructions": {
            "user_diff_detected": "Fill 1 if A/B are perceptibly different, else 0.",
            "user_choice_high_end": "Fill A or B based on which better matches adjective_high. Use 0 if unsure.",
            "user_confidence_1to5": "Optional confidence score.",
        },
    }
    _save_json(out_dir / "perceptual_manifest.json", manifest)

    print(f"Saved: {trial_csv}")
    print(f"Saved: {out_dir / 'perceptual_manifest.json'}")
    print(f"Saved signals: {sig_dir}")


def _truthy_int(x: str) -> int | None:
    s = (x or "").strip().lower()
    if s == "":
        return None
    if s in {"1", "true", "yes", "y"}:
        return 1
    if s in {"0", "false", "no", "n"}:
        return 0
    return None


def summarize(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    with open(args.trial_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    by_pc: dict[str, list[dict[str, str]]] = {}
    for r in rows:
        by_pc.setdefault(r["pc"], []).append(r)

    per_pc = {}
    for pc, pc_rows in sorted(by_pc.items(), key=lambda kv: int(kv[0].replace("PC", ""))):
        diff_answers = [_truthy_int(r.get("user_diff_detected", "")) for r in pc_rows]
        diff_valid = [x for x in diff_answers if x is not None]
        diff_yes = sum(1 for x in diff_valid if x == 1)

        choices = [(r.get("user_choice_high_end", "") or "").strip().upper() for r in pc_rows]
        high_sides = [(r.get("high_side", "") or "").strip().upper() for r in pc_rows]
        dir_pairs = [(c, h) for c, h in zip(choices, high_sides) if c in {"A", "B"} and h in {"A", "B"}]
        dir_correct = sum(1 for c, h in dir_pairs if c == h)

        n_trials = len(pc_rows)
        n_diff_answered = len(diff_valid)
        n_dir_answered = len(dir_pairs)
        perceivability = (diff_yes / n_diff_answered) if n_diff_answered > 0 else 0.0
        direction = (dir_correct / n_dir_answered) if n_dir_answered > 0 else 0.0

        pass_trials = n_diff_answered >= args.min_trials_per_pc and n_dir_answered >= args.min_trials_per_pc
        pass_perc = perceivability >= 0.75
        pass_dir = direction >= 0.70

        if pass_trials and pass_perc and pass_dir:
            status = "PASS"
        elif (perceivability >= 0.65 and direction >= 0.60) and (n_diff_answered >= max(6, args.min_trials_per_pc // 2)):
            status = "WARN"
        else:
            status = "FAIL"

        per_pc[pc] = {
            "status": status,
            "n_trials": n_trials,
            "n_diff_answered": n_diff_answered,
            "n_dir_answered": n_dir_answered,
            "perceivability_rate": round(perceivability, 4),
            "direction_consistency_rate": round(direction, 4),
            "thresholds": {
                "min_trials_per_pc": args.min_trials_per_pc,
                "perceivability_rate": ">=0.75",
                "direction_consistency_rate": ">=0.70",
            },
        }

    n_pass = sum(1 for v in per_pc.values() if v["status"] == "PASS")
    n_warn = sum(1 for v in per_pc.values() if v["status"] == "WARN")
    n_fail = sum(1 for v in per_pc.values() if v["status"] == "FAIL")

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "trial_csv": os.path.abspath(args.trial_csv),
        "summary": {
            "n_pcs": len(per_pc),
            "pass": n_pass,
            "warn": n_warn,
            "fail": n_fail,
        },
        "per_pc": per_pc,
    }
    out_json = out_dir / "perceptual_summary.json"
    _save_json(out_json, summary)

    md = [
        "# Perceptual AB Summary",
        "",
        f"- PASS: {n_pass}",
        f"- WARN: {n_warn}",
        f"- FAIL: {n_fail}",
        "",
        "| PC | Status | Perceivability | Direction | Answered(diff/dir) |",
        "|---|---|---:|---:|---:|",
    ]
    for pc, info in per_pc.items():
        md.append(
            f"| {pc} | {info['status']} | {info['perceivability_rate']:.2%} | "
            f"{info['direction_consistency_rate']:.2%} | "
            f"{info['n_diff_answered']}/{info['n_dir_answered']} |"
        )
    out_md = out_dir / "perceptual_summary.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print(f"Saved: {out_json}")
    print(f"Saved: {out_md}")


def _replace_or_append_validation_section(table_text: str, section: str) -> str:
    marker = "<!-- LABELING_VALIDATION -->"
    if marker in table_text:
        head = table_text.split(marker, 1)[0].rstrip()
        return f"{head}\n\n{section}\n"
    return f"{table_text.rstrip()}\n\n{section}\n"


def freeze(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    controls_spec = _load_json(args.controls_spec)
    objective = _load_json(args.objective_gate)
    perceptual = _load_json(args.perceptual_summary)

    objective_per_pc = objective["per_pc"]
    perceptual_per_pc = perceptual["per_pc"]

    final_by_pc = {}
    for ctrl in controls_spec["controls"]:
        axis = int(ctrl["axis"])
        pc = f"PC{axis + 1}"
        o_status = objective_per_pc.get(pc, {}).get("status", "FAIL")
        p_status = perceptual_per_pc.get(pc, {}).get("status", "FAIL")

        if o_status == "PASS" and p_status == "PASS":
            final = "PASS"
        elif o_status == "FAIL" or p_status == "FAIL":
            final = "FAIL"
        else:
            final = "WARN"

        final_by_pc[pc] = {
            "objective_status": o_status,
            "perceptual_status": p_status,
            "final_status": final,
        }

        ctrl["labeling_validation"] = {
            "objective_status": o_status,
            "perceptual_status": p_status,
            "final_status": final,
            "freeze_recommended": final == "PASS",
            "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        }

    with open(args.controls_spec, "w", encoding="utf-8") as f:
        json.dump(controls_spec, f, indent=2, ensure_ascii=False)

    n_pass = sum(1 for v in final_by_pc.values() if v["final_status"] == "PASS")
    n_warn = sum(1 for v in final_by_pc.values() if v["final_status"] == "WARN")
    n_fail = sum(1 for v in final_by_pc.values() if v["final_status"] == "FAIL")

    report_md = [
        "# Final PC Labeling Report",
        "",
        f"- PASS: {n_pass}",
        f"- WARN: {n_warn}",
        f"- FAIL: {n_fail}",
        "",
        "| PC | Objective | Perceptual | Final |",
        "|---|---|---|---|",
    ]
    for i in range(int(controls_spec["n_controls"])):
        pc = f"PC{i + 1}"
        st = final_by_pc[pc]
        report_md.append(
            f"| {pc} | {st['objective_status']} | {st['perceptual_status']} | {st['final_status']} |"
        )
    report_md.append("")

    # Include criteria excerpt for traceability.
    report_md.append("## Objective Criteria Excerpt")
    report_md.append("")
    for i in range(int(controls_spec["n_controls"])):
        pc = f"PC{i + 1}"
        report_md.append(f"### {pc}")
        crit = objective_per_pc.get(pc, {}).get("criteria", [])
        for c in crit:
            sign = "PASS" if c.get("pass") else "FAIL"
            report_md.append(
                f"- {sign} `{c.get('name')}`: value={c.get('value')} threshold={c.get('threshold')}"
            )
        p = perceptual_per_pc.get(pc, {})
        report_md.append(
            f"- Perceptual: perceivability={p.get('perceivability_rate')} "
            f"direction={p.get('direction_consistency_rate')} status={p.get('status')}"
        )
        report_md.append("")

    report_path = out_dir / "pc_labeling_final_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_md))

    # Update controls_table_v2.md (append validation snapshot section).
    section = [
        "<!-- LABELING_VALIDATION -->",
        "## Labeling Validation Snapshot",
        "",
        "| PC | Objective | Perceptual | Final |",
        "|---|---|---|---|",
    ]
    for i in range(int(controls_spec["n_controls"])):
        pc = f"PC{i + 1}"
        st = final_by_pc[pc]
        section.append(
            f"| {pc} | {st['objective_status']} | {st['perceptual_status']} | {st['final_status']} |"
        )
    section_text = "\n".join(section)

    with open(args.controls_table, "r", encoding="utf-8") as f:
        original_table = f.read()
    updated_table = _replace_or_append_validation_section(original_table, section_text)
    with open(args.controls_table, "w", encoding="utf-8") as f:
        f.write(updated_table)

    print(f"Updated: {args.controls_spec}")
    print(f"Updated: {args.controls_table}")
    print(f"Saved: {report_path}")


def main() -> None:
    args = parse_args()
    if args.mode == "generate":
        generate(args)
    elif args.mode == "summarize":
        summarize(args)
    elif args.mode == "freeze":
        freeze(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
