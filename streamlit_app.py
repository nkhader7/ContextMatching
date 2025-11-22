"""Streamlit application for matching policy statements to control descriptions."""

from __future__ import annotations

import os
from typing import List, Tuple

import requests

import numpy as np
import pandas as pd
import streamlit as st


LLM_API_URL = "https://your-llm-endpoint/v1/chat/completions"
LLM_API_KEY = "your-llm-api-key"
MODEL_NAME = "mistral-large"


def call_llm(text: str) -> dict:
    """Call a self-hosted Mistral LLM with hard-coded configuration.

    Update `LLM_API_URL`, `LLM_API_KEY`, and `MODEL_NAME` with your deployment
    details. The function sends a chat-style request and returns the full JSON
    response payload so callers can reuse the content for embeddings or other
    downstream logic.
    """

    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": text}],
        "temperature": 0.2,
    }

    try:
        response = requests.post(LLM_API_URL, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
    except requests.RequestException as exc:  # pragma: no cover - runtime safeguard
        raise RuntimeError(f"LLM request failed: {exc}") from exc

    try:
        data = response.json()
        # Return the full JSON payload so callers can inspect metadata or
        # re-use the generated content for embedding logic.
        return data
    except (KeyError, IndexError, TypeError) as exc:  # pragma: no cover - defensive
        raise RuntimeError("Unexpected LLM response format") from exc


def _require_mistral_client():
    try:
        from mistralai.client import MistralClient
    except ImportError as exc:  # pragma: no cover - runtime safeguard
        raise ImportError(
            "mistralai package is required for embedding generation. "
            "Install it with `pip install mistralai`."
        ) from exc

    return MistralClient


def generate_mistral_embeddings(texts: List[str]) -> np.ndarray:
    """Generate embeddings for a list of texts using Mistral embeddings."""
    if not texts:
        return np.empty((0, 0))

    api_key = os.getenv("MISTRAL_API_KEY")
    model = os.getenv("MISTRAL_EMBED_MODEL", "mistral-embed")
    endpoint = os.getenv("MISTRAL_ENDPOINT")

    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY environment variable is required for embeddings.")

    MistralClient = _require_mistral_client()
    client = MistralClient(api_key=api_key, endpoint=endpoint)

    response = client.embeddings(model=model, input=texts)
    embeddings = np.array([item.embedding for item in response.data])
    return embeddings


def _cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between two embedding matrices."""
    if a.size == 0 or b.size == 0:
        return np.zeros((len(a), len(b)))

    # Normalize rows
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.matmul(a_norm, b_norm.T)


def find_best_matches(
    policy_df: pd.DataFrame, control_df: pd.DataFrame, threshold: float
) -> Tuple[pd.DataFrame, bool]:
    """Attach best matching control to each policy statement."""
    required_policy_cols = {"policy", "standard_name", "statement"}
    required_control_cols = {"ID", "description", "status", "name"}

    if not required_policy_cols.issubset(policy_df.columns):
        missing = required_policy_cols - set(policy_df.columns)
        raise ValueError(f"Policy file missing columns: {', '.join(sorted(missing))}")

    if not required_control_cols.issubset(control_df.columns):
        missing = required_control_cols - set(control_df.columns)
        raise ValueError(f"Control file missing columns: {', '.join(sorted(missing))}")

    policy_texts = policy_df["statement"].astype(str).tolist()
    control_texts = control_df["description"].astype(str).tolist()

    policy_embeddings = generate_mistral_embeddings(policy_texts)
    control_embeddings = generate_mistral_embeddings(control_texts)

    similarity_matrix = _cosine_similarity_matrix(policy_embeddings, control_embeddings)

    matches = []
    for idx, statement in enumerate(policy_df.itertuples(index=False)):
        row_similarities = similarity_matrix[idx]
        if row_similarities.size == 0:
            best_idx = None
            best_score = 0.0
        else:
            best_idx = int(np.argmax(row_similarities))
            best_score = float(row_similarities[best_idx])

        if best_idx is not None and best_score >= threshold:
            best_match = control_df.iloc[best_idx]
            matches.append(
                {
                    "ID": best_match.ID,
                    "description": best_match.description,
                    "status": best_match.status,
                    "name": best_match.name,
                    "match_percentage": round(best_score * 100, 2),
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


def _load_uploaded_table(uploaded_file) -> pd.DataFrame:
    """Load a CSV or Excel file uploaded via Streamlit."""

    if uploaded_file is None:
        raise ValueError("No file provided")

    filename = uploaded_file.name.lower()
    try:
        if filename.endswith(".csv"):
            return pd.read_csv(uploaded_file)
        if filename.endswith(".xlsx"):
            return pd.read_excel(uploaded_file, engine="openpyxl")
        if filename.endswith(".xls"):
            return pd.read_excel(uploaded_file, engine="xlrd")
    except Exception as exc:  # pragma: no cover - user feedback path
        raise ValueError(f"Failed to read {uploaded_file.name}: {exc}") from exc

    raise ValueError("Unsupported file type. Please upload CSV or Excel files.")


def main():
    st.set_page_config(page_title="Policy-Control Matching", layout="wide")
    st.title("Policy to Control Matching")
    st.write(
        "Upload policy statements and control descriptions to compute similarity using Mistral embeddings."
    )

    threshold = st.slider(
        "Similarity threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.6,
        step=0.01,
        help="Only matches at or above this score will be retained.",
    )

    col1, col2 = st.columns(2)
    with col1:
        policy_file = st.file_uploader(
            "Upload Policy File (CSV or Excel with policy, standard_name, statement)",
            type=["csv", "xlsx", "xls"],
        )
    with col2:
        control_file = st.file_uploader(
            "Upload Control File (CSV or Excel with ID, description, status, name)",
            type=["csv", "xlsx", "xls"],
        )

    policy_df = None
    control_df = None

    if policy_file:
        try:
            policy_df = _load_uploaded_table(policy_file)
            st.subheader("Policy File Preview")
            st.dataframe(policy_df.head())
        except Exception as exc:  # pragma: no cover - user feedback path
            st.error(str(exc))

    if control_file:
        try:
            control_df = _load_uploaded_table(control_file)
            st.subheader("Control File Preview")
            st.dataframe(control_df.head())
        except Exception as exc:  # pragma: no cover - user feedback path
            st.error(str(exc))

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
