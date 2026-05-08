from __future__ import annotations

import itertools
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


DATA_DIR = Path(r"C:\Users\11604\Downloads\user study data")
OUT_DIR = Path(__file__).resolve().parent / "results_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FLOW_ORDER = ["success", "error", "notification", "loading"]
FLOW_LABELS = {
    "success": "Payment\nConfirmation",
    "error": "Failed\nSubmission",
    "notification": "Notification\nReceived",
    "loading": "Pull to\nRefresh",
}
METHOD_ORDER = [
    "random_noise",
    "llm_direct",
    "hapticgen",
    "random_vae_vector",
    "semantic_vae",
]
METHOD_LABELS = {
    "random_noise": "Random Noise",
    "random_vae_vector": "Random-Plan VAE",
    "hapticgen": "HapticGen",
    "llm_direct": "Direct LLM Pattern",
    "semantic_vae": "Semantic VAE",
}
METHOD_COLORS = {
    "Random Noise": "#8B8B8B",
    "Direct LLM Pattern": "#C96B5A",
    "HapticGen": "#4E79A7",
    "Random-Plan VAE": "#59A14F",
    "Semantic VAE": "#7B61A8",
}


def read_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ratings = pd.read_csv(DATA_DIR / "block_ratings.csv")
    key = pd.read_csv(DATA_DIR / "stimulus_key.csv")
    overview = pd.read_csv(DATA_DIR / "flow_overview.csv")
    survey = pd.read_csv(DATA_DIR / "Cindy Thesis Study (Responses) - Form Responses 1.csv")
    return ratings, key, overview, survey


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["method_label"] = df["method"].map(METHOD_LABELS).fillna(df["method"])
    df["flow_label"] = df["flow"].map(FLOW_LABELS).fillna(df["flow"])
    return df


def sem(series: pd.Series) -> float:
    series = series.dropna()
    if len(series) <= 1:
        return float("nan")
    return float(series.std(ddof=1) / math.sqrt(len(series)))


def ci95(series: pd.Series) -> float:
    series = series.dropna()
    if len(series) <= 1:
        return float("nan")
    return float(1.96 * sem(series))


def holm_adjust(p_values: list[float]) -> list[float]:
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [np.nan] * len(p_values)
    running_max = 0.0
    m = len(p_values)
    for rank, (idx, p) in enumerate(indexed, start=1):
        adj = min((m - rank + 1) * p, 1.0)
        running_max = max(running_max, adj)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted


def friedman_and_pairwise(df: pd.DataFrame, value_col: str, group_cols: list[str], label: str) -> pd.DataFrame:
    rows = []
    grouped = [((), df)] if not group_cols else df.groupby(group_cols, dropna=False)
    for group_values, g in grouped:
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        pivot = g.pivot_table(index="participant_id", columns="method", values=value_col, aggfunc="mean")
        pivot = pivot.reindex(columns=METHOD_ORDER).dropna()
        if len(pivot) < 2:
            continue
        arrays = [pivot[m].to_numpy() for m in METHOD_ORDER]
        statistic, p_value = stats.friedmanchisquare(*arrays)
        base = dict(zip(group_cols, group_values))
        rows.append(
            {
                **base,
                "comparison": "friedman",
                "metric": label,
                "method_a": "",
                "method_b": "",
                "n_participants": len(pivot),
                "statistic": statistic,
                "p_value": p_value,
                "p_holm": np.nan,
            }
        )
        pair_rows = []
        p_values = []
        for a, b in itertools.combinations(METHOD_ORDER, 2):
            diffs = pivot[a] - pivot[b]
            try:
                stat, p = stats.wilcoxon(pivot[a], pivot[b], zero_method="wilcox")
            except ValueError:
                stat, p = np.nan, 1.0
            p_values.append(p)
            pair_rows.append(
                {
                    **base,
                    "comparison": "wilcoxon_pairwise",
                    "metric": label,
                    "method_a": METHOD_LABELS[a],
                    "method_b": METHOD_LABELS[b],
                    "n_participants": len(pivot),
                    "statistic": stat,
                    "p_value": p,
                    "median_difference_a_minus_b": float(np.median(diffs)),
                }
            )
        adjusted = holm_adjust(p_values)
        for row, p_holm in zip(pair_rows, adjusted):
            rows.append({**row, "p_holm": p_holm})
    return pd.DataFrame(rows)


def split_ids(value) -> list[str]:
    if pd.isna(value) or str(value).strip() == "":
        return []
    return [item.strip() for item in str(value).split("|") if item.strip()]


def summarize_ratings(ratings: pd.DataFrame, key: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    key_small = key[["anonymous_id", "flow", "method", "variant"]].copy()
    merged = ratings.merge(key_small, on=["anonymous_id", "flow"], how="left", validate="many_to_one")
    merged = add_labels(merged)
    merged["rating_match"] = pd.to_numeric(merged["rating_match"], errors="coerce")
    merged["rating_meaningful"] = pd.to_numeric(merged["rating_meaningful"], errors="coerce")
    merged["rating_average"] = merged[["rating_match", "rating_meaningful"]].mean(axis=1)

    participant = (
        merged.groupby(["participant_id", "flow", "flow_label", "method", "method_label"], as_index=False)
        .agg(
            rating_match=("rating_match", "mean"),
            rating_meaningful=("rating_meaningful", "mean"),
            rating_average=("rating_average", "mean"),
            n_variants=("anonymous_id", "count"),
        )
    )

    by_method_flow = (
        participant.groupby(["flow", "flow_label", "method", "method_label"], as_index=False)
        .agg(
            n_participants=("participant_id", "nunique"),
            match_mean=("rating_match", "mean"),
            match_sd=("rating_match", "std"),
            match_se=("rating_match", sem),
            match_ci95=("rating_match", ci95),
            meaningful_mean=("rating_meaningful", "mean"),
            meaningful_sd=("rating_meaningful", "std"),
            meaningful_se=("rating_meaningful", sem),
            meaningful_ci95=("rating_meaningful", ci95),
            average_mean=("rating_average", "mean"),
            average_sd=("rating_average", "std"),
            average_se=("rating_average", sem),
            average_ci95=("rating_average", ci95),
        )
    )

    overall = (
        participant.groupby(["method", "method_label"], as_index=False)
        .agg(
            n_participant_flow_cells=("participant_id", "count"),
            match_mean=("rating_match", "mean"),
            match_sd=("rating_match", "std"),
            match_se=("rating_match", sem),
            match_ci95=("rating_match", ci95),
            meaningful_mean=("rating_meaningful", "mean"),
            meaningful_sd=("rating_meaningful", "std"),
            meaningful_se=("rating_meaningful", sem),
            meaningful_ci95=("rating_meaningful", ci95),
            average_mean=("rating_average", "mean"),
            average_sd=("rating_average", "std"),
            average_se=("rating_average", sem),
            average_ci95=("rating_average", ci95),
        )
    )

    participant.to_csv(OUT_DIR / "participant_method_flow_ratings.csv", index=False)
    by_method_flow.to_csv(OUT_DIR / "rating_summary_by_method_flow.csv", index=False)
    overall.to_csv(OUT_DIR / "rating_summary_overall.csv", index=False)
    merged.to_csv(OUT_DIR / "ratings_with_methods.csv", index=False)
    return participant, by_method_flow, overall


def summarize_selections(overview: pd.DataFrame, key: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    key_lookup = key.copy()
    key_lookup["anonymous_id"] = key_lookup["anonymous_id"].astype(str)
    for _, row in overview.iterrows():
        for selection_type, id_col, reason_col in [
            ("best", "top3_ids", "top_reason"),
            ("worst", "bottom_ids", "bottom_reason"),
        ]:
            for anon_id in split_ids(row[id_col]):
                match = key_lookup[
                    (key_lookup["flow"] == row["flow"]) & (key_lookup["anonymous_id"] == str(anon_id))
                ]
                if match.empty:
                    method = ""
                    variant = ""
                else:
                    method = match.iloc[0]["method"]
                    variant = match.iloc[0]["variant"]
                rows.append(
                    {
                        "participant_id": row["participant_id"],
                        "flow": row["flow"],
                        "selection_type": selection_type,
                        "anonymous_id": anon_id,
                        "method": method,
                        "variant": variant,
                        "reason_text": "" if pd.isna(row[reason_col]) else row[reason_col],
                    }
                )
    selections = add_labels(pd.DataFrame(rows))
    selections.to_csv(OUT_DIR / "best_worst_selections_long.csv", index=False)

    counts = (
        selections.groupby(["selection_type", "flow", "flow_label", "method", "method_label"], as_index=False)
        .size()
        .rename(columns={"size": "selection_count"})
    )
    counts.to_csv(OUT_DIR / "best_worst_counts_by_method_flow.csv", index=False)

    overall = (
        selections.groupby(["selection_type", "method", "method_label"], as_index=False)
        .size()
        .rename(columns={"size": "selection_count"})
    )
    overall.to_csv(OUT_DIR / "best_worst_counts_overall.csv", index=False)

    participant_any = []
    for (pid, flow, sel_type), g in selections.groupby(["participant_id", "flow", "selection_type"]):
        selected_methods = set(g["method"])
        for method in METHOD_ORDER:
            participant_any.append(
                {
                    "participant_id": pid,
                    "flow": flow,
                    "selection_type": sel_type,
                    "method": method,
                    "selected_any": int(method in selected_methods),
                    "selected_count": int((g["method"] == method).sum()),
                }
            )
    participant_selection = add_labels(pd.DataFrame(participant_any))
    participant_selection.to_csv(OUT_DIR / "participant_selection_rates.csv", index=False)
    return selections, counts, overall


def parse_first_number(value) -> float:
    if pd.isna(value):
        return np.nan
    match = re.search(r"[-+]?\d*\.?\d+", str(value))
    return float(match.group(0)) if match else np.nan


def summarize_survey(survey: pd.DataFrame) -> None:
    cols = list(survey.columns)
    participant_col = "Participant ID (P-XXX)"
    safe_cols = [participant_col]
    background_cols = [c for c in cols if "smartphone apps" in c or "notice vibration" in c]
    likert_cols = [
        c
        for c in cols
        if c.startswith("1. I could")
        or c.startswith("2. I noticed")
        or c.startswith("3.  While")
        or c.startswith("4. How confident")
        or c.startswith("5. Some haptic")
    ]
    rank_cols = [c for c in cols if "Please rank" in c]
    text_cols = [c for c in cols if c.startswith("What made") or c.startswith("Any comments")]

    background = []
    for col in background_cols:
        counts = survey[col].value_counts(dropna=False).reset_index()
        counts.columns = ["response", "count"]
        counts.insert(0, "question", col)
        background.append(counts)
    pd.concat(background, ignore_index=True).to_csv(OUT_DIR / "survey_background_counts.csv", index=False)

    likert = survey[[participant_col] + likert_cols].copy()
    for col in likert_cols:
        likert[col] = likert[col].apply(parse_first_number)
    likert_summary = (
        likert[likert_cols]
        .agg(["count", "mean", "std", "median"])
        .transpose()
        .reset_index()
        .rename(columns={"index": "question"})
    )
    likert_summary.to_csv(OUT_DIR / "survey_likert_summary.csv", index=False)

    ranks = survey[[participant_col] + rank_cols].copy()
    for col in rank_cols:
        ranks[col] = ranks[col].apply(parse_first_number)
    rank_summary = (
        ranks[rank_cols]
        .agg(["count", "mean", "std", "median"])
        .transpose()
        .reset_index()
        .rename(columns={"index": "factor"})
    )
    rank_summary["factor"] = rank_summary["factor"].str.extract(r"\[(.*?)\]", expand=False)
    rank_summary.to_csv(OUT_DIR / "survey_factor_rank_summary.csv", index=False)

    text_export = survey[[participant_col] + text_cols].copy()
    text_export.to_csv(OUT_DIR / "survey_open_text_anonymized.csv", index=False)


def ordered(df: pd.DataFrame, by: str = "method") -> pd.DataFrame:
    df = df.copy()
    if by == "method":
        df["_method_order"] = df["method"].map({m: i for i, m in enumerate(METHOD_ORDER)})
        return df.sort_values("_method_order").drop(columns="_method_order")
    df["_flow_order"] = df["flow"].map({f: i for i, f in enumerate(FLOW_ORDER)})
    return df.sort_values("_flow_order").drop(columns="_flow_order")


def plot_overall_ratings(overall: pd.DataFrame) -> None:
    data = ordered(overall)
    x = np.arange(len(data))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    colors = [METHOD_COLORS[label] for label in data["method_label"]]
    ax.bar(x - width / 2, data["meaningful_mean"], width, yerr=data["meaningful_ci95"], label="Meaningfulness", color=colors, alpha=0.82, capsize=3)
    ax.bar(x + width / 2, data["match_mean"], width, yerr=data["match_ci95"], label="Visual-event fit", color=colors, alpha=0.45, capsize=3)
    ax.set_ylim(1, 7)
    ax.set_ylabel("Mean rating (1-7)")
    ax.set_xticks(x)
    ax.set_xticklabels(data["method_label"], rotation=25, ha="right")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_overall_ratings.pdf")
    fig.savefig(OUT_DIR / "fig_overall_ratings.png", dpi=200)
    plt.close(fig)


def plot_flow_ratings(by_method_flow: pd.DataFrame, metric: str, output_name: str, ylabel: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.6), sharey=True)
    axes = axes.flatten()
    for ax, flow in zip(axes, FLOW_ORDER):
        data = ordered(by_method_flow[by_method_flow["flow"] == flow])
        x = np.arange(len(data))
        labels = data["method_label"].tolist()
        colors = [METHOD_COLORS[label] for label in labels]
        means = data[f"{metric}_mean"]
        errs = data[f"{metric}_ci95"]
        ax.bar(x, means, yerr=errs, color=colors, capsize=3)
        ax.set_title(FLOW_LABELS[flow].replace("\n", " "))
        ax.set_ylim(1, 7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel(ylabel)
    axes[2].set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{output_name}.pdf")
    fig.savefig(OUT_DIR / f"{output_name}.png", dpi=200)
    plt.close(fig)


def plot_best_worst(overall: pd.DataFrame) -> None:
    pivot = overall.pivot_table(index=["method", "method_label"], columns="selection_type", values="selection_count", fill_value=0).reset_index()
    pivot = ordered(pivot)
    x = np.arange(len(pivot))
    fig, ax = plt.subplots(figsize=(7.0, 4.1))
    colors = [METHOD_COLORS[label] for label in pivot["method_label"]]
    ax.bar(x, pivot.get("best", 0), label="Best selections", color=colors, alpha=0.82)
    ax.bar(x, -pivot.get("worst", 0), label="Worst selections", color=colors, alpha=0.45)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot["method_label"], rotation=25, ha="right")
    ax.set_ylabel("Selection count")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_best_worst_overall.pdf")
    fig.savefig(OUT_DIR / "fig_best_worst_overall.png", dpi=200)
    plt.close(fig)


def plot_survey_ranks() -> None:
    path = OUT_DIR / "survey_factor_rank_summary.csv"
    if not path.exists():
        return
    data = pd.read_csv(path).sort_values("mean", ascending=False)
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    x = np.arange(len(data))
    ax.bar(x, data["mean"], yerr=1.96 * data["std"] / np.sqrt(data["count"]), color="#4E79A7", capsize=3)
    ax.set_ylim(1, 5)
    ax.set_ylabel("Mean importance rank (1-5)")
    ax.set_xticks(x)
    ax.set_xticklabels(data["factor"], rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_survey_factor_ranks.pdf")
    fig.savefig(OUT_DIR / "fig_survey_factor_ranks.png", dpi=200)
    plt.close(fig)


def write_plain_summary(overall: pd.DataFrame, selection_overall: pd.DataFrame) -> None:
    overall_ordered = ordered(overall)
    pivot = selection_overall.pivot_table(index=["method", "method_label"], columns="selection_type", values="selection_count", fill_value=0).reset_index()
    pivot = ordered(pivot)
    lines = []
    lines.append("Overall participant-flow rating means")
    for _, row in overall_ordered.iterrows():
        lines.append(
            f"- {row['method_label']}: meaningful={row['meaningful_mean']:.2f}, fit={row['match_mean']:.2f}, average={row['average_mean']:.2f}"
        )
    lines.append("")
    lines.append("Overall best/worst selection counts")
    for _, row in pivot.iterrows():
        lines.append(f"- {row['method_label']}: best={int(row.get('best', 0))}, worst={int(row.get('worst', 0))}")
    (OUT_DIR / "analysis_quick_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def write_data_quality_report(
    ratings: pd.DataFrame,
    key: pd.DataFrame,
    overview: pd.DataFrame,
    survey: pd.DataFrame,
    participant: pd.DataFrame,
    selections: pd.DataFrame,
) -> None:
    lines = []
    lines.append("Data quality summary")
    lines.append(f"- block_ratings rows: {len(ratings)}")
    lines.append(f"- flow_overview rows: {len(overview)}")
    lines.append(f"- stimulus_key rows: {len(key)}")
    lines.append(f"- follow-up survey rows: {len(survey)}")
    lines.append(f"- unique participants in ratings: {ratings['participant_id'].nunique()}")
    lines.append(f"- unique participants in flow overview: {overview['participant_id'].nunique()}")
    lines.append(f"- unique participants in follow-up survey: {survey['Participant ID (P-XXX)'].nunique()}")
    lines.append(f"- participant method-flow rating cells: {len(participant)}")
    lines.append(f"- expected participant method-flow cells: {27 * 4 * 5}")
    lines.append(f"- rating missing values: {ratings[['rating_match', 'rating_meaningful']].isna().sum().to_dict()}")
    lines.append(f"- selected best IDs: {int((selections['selection_type'] == 'best').sum())}")
    lines.append(f"- selected worst IDs: {int((selections['selection_type'] == 'worst').sum())}")
    lines.append("")
    lines.append("Stimulus key method counts by flow")
    counts = key.groupby(["flow", "method"]).size().reset_index(name="count")
    for _, row in counts.iterrows():
        lines.append(f"- {row['flow']} / {METHOD_LABELS.get(row['method'], row['method'])}: {row['count']}")
    (OUT_DIR / "data_quality_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ratings, key, overview, survey = read_inputs()
    participant, by_method_flow, overall = summarize_ratings(ratings, key)
    selections, selection_counts, selection_overall = summarize_selections(overview, key)
    summarize_survey(survey)
    write_data_quality_report(ratings, key, overview, survey, participant, selections)

    test_overall_meaningful = friedman_and_pairwise(participant, "rating_meaningful", [], "meaningfulness_overall")
    test_overall_match = friedman_and_pairwise(participant, "rating_match", [], "visual_fit_overall")
    test_flow_meaningful = friedman_and_pairwise(participant, "rating_meaningful", ["flow"], "meaningfulness_by_flow")
    test_flow_match = friedman_and_pairwise(participant, "rating_match", ["flow"], "visual_fit_by_flow")
    pd.concat([test_overall_meaningful, test_overall_match, test_flow_meaningful, test_flow_match], ignore_index=True).to_csv(
        OUT_DIR / "repeated_measures_tests.csv", index=False
    )

    plot_overall_ratings(overall)
    plot_flow_ratings(by_method_flow, "meaningful", "fig_ratings_by_flow_meaningfulness", "Meaningfulness rating (1-7)")
    plot_flow_ratings(by_method_flow, "match", "fig_ratings_by_flow_fit", "Visual-event fit rating (1-7)")
    plot_best_worst(selection_overall)
    plot_survey_ranks()
    write_plain_summary(overall, selection_overall)

    print(f"Wrote analysis outputs to {OUT_DIR}")
    print((OUT_DIR / "analysis_quick_summary.txt").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
