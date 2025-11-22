# ContextMatching

## Overview
This Streamlit application matches policy statements to control descriptions using Mistral embeddings. It uploads two CSV or Excel files, computes cosine similarity scores, and produces a merged dataset with the best match for each policy statement. All LLM calls are routed through a helper that is pre-wired for a self-hosted Mistral-compatible endpoint.

## Prerequisites
- Python 3.9+
- Recommended: virtual environment (e.g., `python -m venv .venv && source .venv/bin/activate`)

## Installation
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
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
- **Policy file (CSV or Excel)** must include columns: `policy`, `standard_name`, `statement`.
- **Control file (CSV or Excel)** must include columns: `ID`, `description`, `status`, `name`.

## Application flow
1. Upload the policy CSV/Excel and control CSV/Excel via the UI.
2. Adjust the similarity threshold slider (0.0–1.0, default 0.60).
3. The app generates Mistral embeddings for policy statements and control descriptions, computes cosine similarity, and finds the best control match per policy statement above the threshold.
4. View previews of both input files and the merged results directly in the UI.
5. Download the merged CSV (includes best match columns `ID`, `description`, `status`, `name`, and `match_percentage`).

### Flow diagram
```mermaid
flowchart TD
    A[Upload policy file (CSV/Excel)] --> B[Upload control file (CSV/Excel)]
    B --> C[Adjust similarity threshold]
    C --> D[Generate Mistral embeddings for statements and descriptions]
    D --> E[Compute cosine similarity matrix]
    E --> F[Select best match per policy above threshold]
    F --> G[Display previews and merged table]
    G --> H[Download merged CSV]
```

### Embedding and matching details
- The app uses the `MistralClient.embeddings` method with the model name from `MISTRAL_EMBED_MODEL` (default `mistral-embed`).
- Cosine similarity is computed between normalized embedding vectors; the highest-scoring control description above the threshold is attached to each policy statement.
- The merged output adds `ID`, `description`, `status`, `name`, and `match_percentage` (0–100, rounded to two decimals).

## Mistral LLM helper
The `call_llm(text: str) -> dict` helper in `streamlit_app.py` is pre-wired to call a self-hosted Mistral-compatible chat endpoint using the hard-coded constants `LLM_API_URL`, `LLM_API_KEY`, and `MODEL_NAME`. Update these values with your deployment details before use. The helper sends the user text as a chat message and returns the full JSON payload (including `choices`, message content, and any metadata), making it easy to reuse the generated text for embedding logic or other downstream processing. A runtime error is raised if the request fails or the response format is unexpected.

## Troubleshooting
- **Missing dependencies**: Ensure `mistralai`, `streamlit`, `pandas`, `numpy`, and the Excel readers (`openpyxl`, `xlrd`) are installed (pulled in via `requirements.txt`).
- **Authentication errors**: Verify `MISTRAL_API_KEY` is set and valid.
- **No matches found**: Lower the similarity threshold or confirm the input data aligns across both files.

## Development notes
Run a quick syntax check:
```bash
python -m py_compile streamlit_app.py
```
