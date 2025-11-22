"""Streamlit application for matching policy statements to control descriptions."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Set, Tuple

import pandas as pd
import requests
import streamlit as st


def call_llm(policy_text: str, control_text: str) -> float:
    """Call a self-hosted LLM to score semantic similarity.

    The endpoint must be provided via the ``LLM_ENDPOINT`` environment variable and
    is expected to return JSON containing a numeric score, either as ``match_percentage``
    (0-100) or ``score`` (0-1 or 0-100). The function normalizes the value to a
    percentage between 0 and 100.
    """

    endpoint = os.getenv("LLM_ENDPOINT")
    if not endpoint:
        raise RuntimeError("LLM_ENDPOINT environment variable is required for LLM calls.")

    response = requests.post(
        endpoint,
        json={"policy_text": policy_text, "control_text": control_text},
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()

    for key in ("match_percentage", "score", "matching_score"):
        if key in data:
            raw_score = data[key]
            break
    else:
        raise ValueError("LLM response did not include a score field.")

    try:
        score_value = float(raw_score)
    except (TypeError, ValueError) as exc:  # pragma: no cover - runtime safeguard
        raise ValueError("LLM score must be numeric.") from exc

    if score_value <= 1:
        score_pct = score_value * 100
    else:
        score_pct = score_value

    return max(0.0, min(100.0, score_pct))


def find_best_matches(
    policy_df: pd.DataFrame, control_df: pd.DataFrame, threshold: float
) -> Tuple[pd.DataFrame, bool]:
    """Attach best matching control to each policy statement using LLM scores."""
    required_policy_cols = {"policy", "standard_name", "statement"}
    required_control_cols = {"ID", "description", "status", "name"}

    if not required_policy_cols.issubset(policy_df.columns):
        missing = required_policy_cols - set(policy_df.columns)
        raise ValueError(f"Policy file missing columns: {', '.join(sorted(missing))}")

    if not required_control_cols.issubset(control_df.columns):
        missing = required_control_cols - set(control_df.columns)
        raise ValueError(f"Control file missing columns: {', '.join(sorted(missing))}")

    matches = []
    for policy_idx, policy_row in enumerate(policy_df.itertuples(index=False)):
        best_idx = None
        best_score = -1.0

        for control_idx, control_row in enumerate(control_df.itertuples(index=False)):
            score = call_llm(str(policy_row.statement), str(control_row.description))
            if score > best_score:
                best_score = score
                best_idx = control_idx

        if best_idx is not None and best_score >= threshold:
            best_match = control_df.iloc[best_idx]
            matches.append(
                {
                    "ID": best_match.ID,
                    "description": best_match.description,
                    "status": best_match.status,
                    "name": best_match.name,
                    "match_percentage": round(best_score, 2),
                }
            )
        else:
            matches.append(
                {
                    "ID": None,
                    "description": None,
                    "status": None,
                    "name": None,
                    "match_percentage": None,
                }
            )

    merged = pd.concat([policy_df.reset_index(drop=True), pd.DataFrame(matches)], axis=1)
    has_matches = any(match["match_percentage"] is not None for match in matches)
    return merged, has_matches


def _load_table(uploaded_file, expected_columns: Set[str], role: str) -> pd.DataFrame:
    """Load a CSV or Excel file and validate required columns."""

    ext = Path(uploaded_file.name).suffix.lower()
    try:
        if ext in {".xls", ".xlsx"}:
            df = pd.read_excel(uploaded_file)
        elif ext == ".csv":
            try:
                df = pd.read_csv(uploaded_file, encoding="utf-8")
            except UnicodeDecodeError:
                uploaded_file.seek(0)
                try:
                    df = pd.read_csv(uploaded_file, encoding="utf-8-sig")
                except UnicodeDecodeError:
                    uploaded_file.seek(0)
                    df = pd.read_csv(
                        uploaded_file, encoding="latin-1", on_bad_lines="skip"
                    )
        else:
            raise ValueError("Unsupported file type. Please upload CSV or Excel files.")
    except Exception as exc:  # pragma: no cover - user feedback path
        raise ValueError(f"Unable to read {role} file: {exc}") from exc

    missing = expected_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"{role} file missing columns: {', '.join(sorted(missing))}"
        )

    return df


def main():
    st.set_page_config(page_title="Policy-Control Matching", layout="wide")
    st.title("Policy to Control Matching")
    st.write(
        "Upload policy statements and control descriptions to compute semantic similarity using a self-hosted LLM."
    )

    threshold = st.slider(
        "Similarity threshold (%)",
        min_value=0,
        max_value=100,
        value=60,
        step=1,
        help="Only matches at or above this percentage will be retained.",
    )

    col1, col2 = st.columns(2)
    with col1:
        policy_file = st.file_uploader(
            "Upload Policy File (CSV or Excel with policy, standard_name, statement)",
            type=["csv", "xls", "xlsx"],
        )
    with col2:
        control_file = st.file_uploader(
            "Upload Control File (CSV or Excel with ID, description, status, name)",
            type=["csv", "xls", "xlsx"],
        )

    policy_df = None
    control_df = None

    if policy_file:
        try:
            policy_df = _load_table(policy_file, {"policy", "standard_name", "statement"}, "Policy")
        except ValueError as exc:  # pragma: no cover - user feedback path
            st.error(str(exc))
        else:
            st.subheader("Policy File Preview")
            st.dataframe(policy_df.head())

    if control_file:
        try:
            control_df = _load_table(
                control_file, {"ID", "description", "status", "name"}, "Control"
            )
        except ValueError as exc:  # pragma: no cover - user feedback path
            st.error(str(exc))
        else:
            st.subheader("Control File Preview")
            st.dataframe(control_df.head())

    st.markdown("---")

    if policy_df is not None and control_df is not None:
        try:
            merged_df, has_matches = find_best_matches(policy_df, control_df, threshold)
        except Exception as exc:  # pragma: no cover - user feedback path
            st.error(str(exc))
            return

        st.subheader("Merged Results")
        st.dataframe(merged_df)

        csv_data = merged_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Merged Results",
            data=csv_data,
            file_name="merged_results.csv",
            mime="text/csv",
        )

        if not has_matches:
            st.info("No matches met the similarity threshold. Adjust the threshold or verify inputs.")
    else:
        st.info("Upload both files to compute matches.")


if __name__ == "__main__":
    main()
