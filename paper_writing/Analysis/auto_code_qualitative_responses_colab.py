# %% [markdown]
# # Qualitative Response Auto-Coding
#
# This notebook-style script extracts open-text responses from the user study,
# applies a codebook-based auto-coding pass, and writes theme-count summaries.
#
# In Google Colab, upload:
# - `flow_overview.csv`
# - `stimulus_key.csv`
# - `qualitative_codebook.csv`
#
# Default mode uses keyword-assisted coding only. Set `USE_LLM = True` to add
# an OpenAI structured-output coding pass.

# %%
import json
import os
import re
from pathlib import Path

import pandas as pd

# In Colab, upload all three CSV files to `/content`.
# Locally, this falls back to the study-data download folder and the codebook
# stored next to this script.
SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
COLAB_DIR = Path("/content")
LOCAL_STUDY_DATA_DIR = Path(r"C:\Users\11604\Downloads\user study data")

if (COLAB_DIR / "flow_overview.csv").exists():
    DATA_DIR = COLAB_DIR
elif (LOCAL_STUDY_DATA_DIR / "flow_overview.csv").exists():
    DATA_DIR = LOCAL_STUDY_DATA_DIR
else:
    DATA_DIR = SCRIPT_DIR

FLOW_OVERVIEW_PATH = DATA_DIR / "flow_overview.csv"
STIMULUS_KEY_PATH = DATA_DIR / "stimulus_key.csv"
CODEBOOK_PATH = DATA_DIR / "qualitative_codebook.csv"
if not CODEBOOK_PATH.exists():
    CODEBOOK_PATH = SCRIPT_DIR / "qualitative_codebook.csv"

OUTPUT_DIR = SCRIPT_DIR / "qualitative_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

USE_LLM = False
OPENAI_MODEL = "gpt-4.1-mini"

METHOD_LABELS = {
    "random_noise": "Random Noise",
    "random_vae_vector": "Random-Plan VAE",
    "hapticgen": "HapticGen",
    "llm_direct": "Direct LLM Pattern",
    "semantic_vae": "Semantic VAE",
}

# %%
flow_overview = pd.read_csv(FLOW_OVERVIEW_PATH)
stimulus_key = pd.read_csv(STIMULUS_KEY_PATH)
codebook = pd.read_csv(CODEBOOK_PATH)

stimulus_key["anonymous_id"] = stimulus_key["anonymous_id"].astype(str)
stimulus_key["method_label"] = stimulus_key["method"].map(METHOD_LABELS).fillna(stimulus_key["method"])

print(flow_overview.head())
print(stimulus_key.head())
print(codebook[["code_id", "theme"]])

# %%
def split_ids(value):
    if pd.isna(value) or str(value).strip() == "":
        return []
    return [item.strip() for item in str(value).split("|") if item.strip()]


reason_groups = []
for _, row in flow_overview.iterrows():
    for selection_type, ids_col, reason_col in [
        ("best", "top3_ids", "top_reason"),
        ("worst", "bottom_ids", "bottom_reason"),
    ]:
        text = "" if pd.isna(row.get(reason_col)) else str(row.get(reason_col)).strip()
        ids = split_ids(row.get(ids_col))
        if text or ids:
            reason_groups.append(
                {
                    "participant_id": row["participant_id"],
                    "flow": row["flow"],
                    "selection_type": selection_type,
                    "selected_ids": "|".join(ids),
                    "reason_text": text,
                }
            )

reason_groups = pd.DataFrame(reason_groups)
reason_groups.to_csv(OUTPUT_DIR / "qualitative_reason_groups_extracted.csv", index=False)
reason_groups.head()

# %%
exploded_rows = []
for _, row in reason_groups.iterrows():
    for anon_id in split_ids(row["selected_ids"]):
        key_match = stimulus_key[
            (stimulus_key["anonymous_id"] == str(anon_id))
            & (stimulus_key["flow"] == row["flow"])
        ]
        if key_match.empty:
            method = ""
            method_label = ""
            variant = ""
        else:
            key_row = key_match.iloc[0]
            method = key_row["method"]
            method_label = key_row["method_label"]
            variant = key_row["variant"]
        exploded_rows.append(
            {
                **row.to_dict(),
                "anonymous_id": str(anon_id),
                "method": method,
                "method_label": method_label,
                "variant": variant,
            }
        )

reason_by_stimulus = pd.DataFrame(exploded_rows)
reason_by_stimulus.to_csv(OUTPUT_DIR / "qualitative_reason_by_selected_stimulus.csv", index=False)
reason_by_stimulus.head()

# %%
def normalize_text(text):
    text = "" if pd.isna(text) else str(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_keywords(keyword_cell):
    if pd.isna(keyword_cell):
        return []
    return [kw.strip().lower() for kw in str(keyword_cell).split(";") if kw.strip()]


keyword_map = {
    row["code_id"]: parse_keywords(row["keyword_hints"])
    for _, row in codebook.iterrows()
}


def keyword_code(text):
    normalized = normalize_text(text)
    codes = []
    matched_terms = {}
    for code_id, keywords in keyword_map.items():
        hits = [kw for kw in keywords if kw and kw in normalized]
        if hits:
            codes.append(code_id)
            matched_terms[code_id] = "|".join(sorted(set(hits)))
    if not codes:
        codes = ["uncoded"]
    return codes, matched_terms


keyword_coded = []
for _, row in reason_by_stimulus.iterrows():
    codes, matched_terms = keyword_code(row["reason_text"])
    for code_id in codes:
        keyword_coded.append(
            {
                **row.to_dict(),
                "code_id": code_id,
                "coding_source": "keyword",
                "matched_terms": matched_terms.get(code_id, ""),
                "confidence": "",
                "rationale": "",
            }
        )

keyword_coded = pd.DataFrame(keyword_coded)
keyword_coded.to_csv(OUTPUT_DIR / "qualitative_coded_keyword.csv", index=False)
keyword_coded.head()

# %%
def summarize_counts(coded_df, source_name):
    counts = (
        coded_df.groupby(["selection_type", "flow", "method_label", "code_id"])
        .size()
        .reset_index(name="count")
        .sort_values(["selection_type", "flow", "method_label", "count"], ascending=[True, True, True, False])
    )
    counts.to_csv(OUTPUT_DIR / f"theme_counts_by_method_flow_{source_name}.csv", index=False)

    overall = (
        coded_df.groupby(["selection_type", "code_id"])
        .size()
        .reset_index(name="count")
        .sort_values(["selection_type", "count"], ascending=[True, False])
    )
    overall.to_csv(OUTPUT_DIR / f"theme_counts_overall_{source_name}.csv", index=False)
    return counts, overall


keyword_counts, keyword_overall = summarize_counts(keyword_coded, "keyword")
keyword_counts.head(20)

# %% [markdown]
# ## Optional OpenAI-assisted coding
#
# This pass asks a model to assign one or more codebook codes to each response.
# Use it to supplement the keyword pass, especially for short or indirect
# responses. Keep a manual spot check for thesis reporting.

# %%
def make_codebook_prompt(codebook_df):
    lines = []
    for _, row in codebook_df.iterrows():
        lines.append(
            f"- {row['code_id']}: {row['theme']}. Definition: {row['definition']} "
            f"Include when: {row['include_when']} Exclude when: {row['exclude_when']}"
        )
    return "\n".join(lines)


def llm_code_responses(reason_df, codebook_df):
    from openai import OpenAI

    client = OpenAI()
    allowed_codes = codebook_df["code_id"].tolist() + ["uncoded"]
    codebook_text = make_codebook_prompt(codebook_df)

    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "codes": {
                "type": "array",
                "items": {"type": "string", "enum": allowed_codes},
            },
            "confidence": {"type": "number"},
            "rationale": {"type": "string"},
        },
        "required": ["codes", "confidence", "rationale"],
    }

    outputs = []
    for idx, row in reason_df.iterrows():
        prompt = f"""
You are coding open-text responses from a haptic feedback user study.
Assign one or more codebook codes to the response. Use "uncoded" only if no code fits.

Codebook:
{codebook_text}

Context:
- flow: {row['flow']}
- selection type: {row['selection_type']}
- method: {row['method_label']}
- response: {row['reason_text']}

Return only the structured JSON object.
"""
        response = client.responses.create(
            model=OPENAI_MODEL,
            input=[
                {"role": "system", "content": "You are a careful qualitative research coding assistant."},
                {"role": "user", "content": prompt},
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "qualitative_codes",
                    "strict": True,
                    "schema": schema,
                }
            },
        )
        parsed = json.loads(response.output_text)
        codes = [code for code in parsed["codes"] if code in allowed_codes]
        if not codes:
            codes = ["uncoded"]
        for code_id in codes:
            outputs.append(
                {
                    **row.to_dict(),
                    "code_id": code_id,
                    "coding_source": "llm",
                    "matched_terms": "",
                    "confidence": parsed.get("confidence", ""),
                    "rationale": parsed.get("rationale", ""),
                }
            )
        if (idx + 1) % 20 == 0:
            print(f"Coded {idx + 1}/{len(reason_df)} rows")

    return pd.DataFrame(outputs)


if USE_LLM:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY before running LLM coding.")
    llm_coded = llm_code_responses(reason_by_stimulus, codebook)
    llm_coded.to_csv(OUTPUT_DIR / "qualitative_coded_llm.csv", index=False)
    llm_counts, llm_overall = summarize_counts(llm_coded, "llm")
    display(llm_counts.head(20))
else:
    print("USE_LLM is False. Keyword-coded outputs were generated.")

# %%
print("Wrote outputs to:", OUTPUT_DIR)
print("\nMain files:")
for path in sorted(OUTPUT_DIR.glob("*.csv")):
    print("-", path.name)
