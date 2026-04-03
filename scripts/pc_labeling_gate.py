"""Objective gate for 8-PC labeling.

Consumes existing validation artifacts and emits per-PC PASS/WARN/FAIL.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Objective gating for PC labeling")
    p.add_argument(
        "--binding_json",
        type=str,
        default="outputs/validation/metric_binding_extended.json",
        help="Path to metric_binding_extended.json",
    )
    p.add_argument(
        "--alignment_json",
        type=str,
        default="outputs/validation/pca_axis_alignment.json",
        help="Path to pca_axis_alignment.json",
    )
    p.add_argument(
        "--cross_seed_json",
        type=str,
        default="outputs/cross_seed/cross_seed_report.json",
        help="Path to cross_seed_report.json",
    )
    p.add_argument("--output_dir", type=str, default="outputs/labeling")
    p.add_argument("--n_pcs", type=int, default=8)
    return p.parse_args()


def _load_json(path: str, required: bool = True) -> dict[str, Any] | None:
    if not os.path.exists(path):
        if required:
            raise FileNotFoundError(path)
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _criterion(name: str, value: Any, threshold: Any, passed: bool, detail: str) -> dict[str, Any]:
    return {
        "name": name,
        "value": value,
        "threshold": threshold,
        "pass": bool(passed),
        "detail": detail,
    }


def _pc_group(idx0: int) -> str:
    return "primary" if idx0 < 4 else "secondary"


def _alignment_threshold(idx0: int) -> float:
    return 0.75 if idx0 < 4 else 0.60


def _effect_threshold(idx0: int) -> float:
    return 0.30 if idx0 < 4 else 0.15


def _nonpeak_binding_threshold(idx0: int) -> tuple[int, float]:
    if idx0 < 4:
        return 2, 0.60
    return 1, 0.50


def _pc_status(criteria: list[dict[str, Any]]) -> str:
    failed = [c for c in criteria if not c["pass"]]
    if not failed:
        return "PASS"

    critical_names = {"cross_seed_alignment", "cross_seed_sign_consistency"}
    if any(c["name"] in critical_names for c in failed):
        return "FAIL"
    if len(failed) <= 2:
        return "WARN"
    return "FAIL"


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    binding = _load_json(args.binding_json, required=True)
    alignment = _load_json(args.alignment_json, required=False)
    cross_seed = _load_json(args.cross_seed_json, required=False)

    ranked_bindings: dict[str, list[dict[str, Any]]] = binding["ranked_bindings"]
    ref_per_pc: dict[str, dict[str, dict[str, Any]]] = binding["reference_comparison"]["per_pc"]
    cross_influence: dict[str, dict[str, Any]] = binding["cross_influence"]
    effect_sizes: dict[str, dict[str, dict[str, float]]] = binding["effect_sizes"]

    align_per_pc = alignment.get("per_pc_avg_alignment", []) if alignment else []
    seed_sign_consistency = cross_seed.get("sign_consistency_per_pc", []) if cross_seed else []

    per_pc = {}
    for idx in range(args.n_pcs):
        pc = f"PC{idx + 1}"
        group = _pc_group(idx)
        min_metrics, rho_thr = _nonpeak_binding_threshold(idx)

        pc_ranked = ranked_bindings.get(pc, [])
        valid_nonpeak = [
            m
            for m in pc_ranked
            if m.get("metric") != "peak_amplitude"
            and float(m.get("p_value", 1.0)) < 0.05
            and abs(float(m.get("rho", 0.0))) >= rho_thr
        ]
        c1 = _criterion(
            "nonpeak_monotonic_metrics",
            len(valid_nonpeak),
            f">={min_metrics} metrics with |rho|>={rho_thr} and p<0.05",
            len(valid_nonpeak) >= min_metrics,
            "Top candidates: "
            + ", ".join(
                f"{m['metric']}({float(m['rho']):+.2f})" for m in valid_nonpeak[:4]
            ),
        )

        ref_metrics = ref_per_pc.get(pc, {})
        focus = [
            v
            for v in ref_metrics.values()
            if abs(float(v.get("rho_origin", 0.0))) >= 0.3
            or abs(float(v.get("rho_mean", 0.0))) >= 0.3
        ]
        if focus:
            sign_consistency = sum(1 for v in focus if bool(v.get("sign_consistent"))) / len(focus)
        else:
            sign_consistency = 0.0
        c2 = _criterion(
            "dual_reference_sign_consistency",
            round(sign_consistency, 4),
            ">=0.80",
            sign_consistency >= 0.80,
            f"Compared metrics: {len(focus)}",
        )

        sel = float(cross_influence.get(pc, {}).get("selectivity", 0.0))
        c3 = _criterion(
            "selectivity",
            round(sel, 4),
            ">=1.50",
            sel >= 1.50,
            "On-target / off-target influence ratio",
        )

        # Primary effect metric is first valid non-peak metric from ranked list.
        chosen_metric = valid_nonpeak[0]["metric"] if valid_nonpeak else None
        rel_change = 0.0
        if chosen_metric and pc in effect_sizes and chosen_metric in effect_sizes[pc]:
            rel_change = float(effect_sizes[pc][chosen_metric]["relative_change"])
        eff_thr = _effect_threshold(idx)
        c4 = _criterion(
            "effect_size_relative_change",
            round(rel_change, 4),
            f">={eff_thr:.2f}",
            rel_change >= eff_thr,
            f"Metric: {chosen_metric or 'N/A'}",
        )

        align_val = float(align_per_pc[idx]) if idx < len(align_per_pc) else 0.0
        align_thr = _alignment_threshold(idx)
        c5 = _criterion(
            "cross_seed_alignment",
            round(align_val, 4),
            f">={align_thr:.2f}",
            align_val >= align_thr,
            "Per-PC cosine alignment across seed PCA components",
        )

        sign_val = float(seed_sign_consistency[idx]) if idx < len(seed_sign_consistency) else 0.0
        c6 = _criterion(
            "cross_seed_sign_consistency",
            round(sign_val, 4),
            ">=0.60",
            sign_val >= 0.60,
            "Fraction of metrics with stable sign across seeds",
        )

        criteria = [c1, c2, c3, c4, c5, c6]
        status = _pc_status(criteria)
        per_pc[pc] = {
            "group": group,
            "status": status,
            "criteria": criteria,
            "selected_primary_metric": chosen_metric,
        }

    n_pass = sum(1 for v in per_pc.values() if v["status"] == "PASS")
    n_warn = sum(1 for v in per_pc.values() if v["status"] == "WARN")
    n_fail = sum(1 for v in per_pc.values() if v["status"] == "FAIL")

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "binding_json": os.path.abspath(args.binding_json),
            "alignment_json": os.path.abspath(args.alignment_json),
            "cross_seed_json": os.path.abspath(args.cross_seed_json),
        },
        "thresholds": {
            "primary_pc_nonpeak": "|rho|>=0.60, p<0.05, at least 2 metrics",
            "secondary_pc_nonpeak": "|rho|>=0.50, p<0.05, at least 1 metric",
            "dual_reference_sign_consistency": ">=0.80",
            "selectivity": ">=1.50",
            "primary_effect_size": ">=0.30",
            "secondary_effect_size": ">=0.15",
            "primary_alignment": ">=0.75",
            "secondary_alignment": ">=0.60",
            "cross_seed_sign_consistency": ">=0.60",
        },
        "summary": {
            "n_pcs": args.n_pcs,
            "pass": n_pass,
            "warn": n_warn,
            "fail": n_fail,
            "freeze_ready": n_pass == args.n_pcs,
        },
        "per_pc": per_pc,
    }

    out_json = out_dir / "pc_labeling_objective_gate.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    md = [
        "# PC Labeling Objective Gate",
        "",
        f"- PASS: {n_pass}",
        f"- WARN: {n_warn}",
        f"- FAIL: {n_fail}",
        f"- Freeze-ready: **{report['summary']['freeze_ready']}**",
        "",
        "| PC | Group | Status | Notes |",
        "|---|---|---|---|",
    ]
    for i in range(args.n_pcs):
        pc = f"PC{i + 1}"
        info = per_pc[pc]
        failed = [c for c in info["criteria"] if not c["pass"]]
        note = "All objective criteria passed." if not failed else "; ".join(
            f"{c['name']}={c['value']} ({c['threshold']})" for c in failed[:3]
        )
        md.append(f"| {pc} | {info['group']} | {info['status']} | {note} |")
    md.append("")

    out_md = out_dir / "pc_labeling_objective_gate.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print(f"Saved: {out_json}")
    print(f"Saved: {out_md}")


if __name__ == "__main__":
    main()
