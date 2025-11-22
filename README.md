# ContextMatching

## Overview
This Streamlit application matches policy statements to control descriptions using Mistral embeddings. It uploads two CSV files, computes cosine similarity scores, and produces a merged dataset with the best match for each policy statement.

## Prerequisites
- Python 3.9+
- Recommended: virtual environment (e.g., `python -m venv .venv && source .venv/bin/activate`)

## Installation
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   If a `requirements.txt` file is not present, install the packages directly:
   ```bash
   pip install streamlit pandas numpy mistralai
   ```

2. Configure environment variables for Mistral embeddings:
   - `MISTRAL_API_KEY` (required): API key for Mistral embeddings.
   - `MISTRAL_ENDPOINT` (optional): Override the default Mistral API endpoint when self-hosting.
   - `MISTRAL_EMBED_MODEL` (optional): Embedding model name (defaults to `mistral-embed`).

## Running the app
From the repository root, launch the Streamlit server:
```bash
streamlit run streamlit_app.py
```
By default, Streamlit runs at http://localhost:8501.

## File formats
- **Policy file (CSV)** must include columns: `policy`, `standard_name`, `statement`.
- **Control file (CSV)** must include columns: `ID`, `description`, `status`, `name`.

## Application flow
1. Upload the policy CSV and control CSV via the UI.
2. Adjust the similarity threshold slider (0.0–1.0, default 0.60).
3. The app generates Mistral embeddings for policy statements and control descriptions, computes cosine similarity, and finds the best control match per policy statement above the threshold.
4. View previews of both input files and the merged results directly in the UI.
5. Download the merged CSV (includes best match columns `ID`, `description`, `status`, `name`, and `match_percentage`).

## Mistral LLM placeholder
The `call_llm(text: str) -> str` helper in `streamlit_app.py` is intentionally unimplemented. Replace the TODO with your self-hosted Mistral LLM invocation when ready.

## Troubleshooting
- **Missing dependencies**: Ensure `mistralai`, `streamlit`, `pandas`, and `numpy` are installed.
- **Authentication errors**: Verify `MISTRAL_API_KEY` is set and valid.
- **No matches found**: Lower the similarity threshold or confirm the input data aligns across both files.

## Development notes
Run a quick syntax check:
```bash
python -m py_compile streamlit_app.py
```
