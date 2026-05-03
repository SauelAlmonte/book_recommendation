# Developer guide

Local setup, data, testing, and deployment notes for the **book-recommendation** backend. For a high-level overview and portfolio context, see the root [README](../README.md).

## Project layout

| Path | Purpose |
|------|---------|
| [`src/book_recommendation/`](../src/book_recommendation/) | FastAPI app and recommendation logic |
| [`tests/`](../tests/) | Pytest suite |
| [`notebooks/`](../notebooks/) | Jupyter workflows (paths assume repo root for CSV outputs) |
| [`static/cover-not-found.jpg`](../static/cover-not-found.jpg) | Fallback cover when a thumbnail is missing |
| [`pyproject.toml`](../pyproject.toml) | API dependencies and packaging |
| [`requirements-notebooks-full.txt`](../requirements-notebooks-full.txt) | Optional large dependency set for notebooks |
| [`tests/fixtures/live_minimal/`](../tests/fixtures/live_minimal/) | Tiny catalog for optional live OpenAI + Chroma tests |

## Catalog file contract

The books CSV must include at least one data row and these columns (exact names):

`isbn13`, `thumbnail`, `simple_categories`, `authors`, `description`, `title`, `joy`, `surprise`, `anger`, `fear`, `sadness`

The tagged description file must be UTF-8 text with **one book per line**. Each non-empty line must **start with the same `isbn13` value** as in the CSV (digits only in the first token), followed by the rest of the content used for embeddings (for example a space and quoted text). Blank lines are ignored until the first non-empty line, which is checked for a leading numeric ISBN.

## Python version

Use **Python 3.10, 3.11, or 3.12** for the most reliable installs. Newer interpreters (for example 3.14) may hit build issues on transitive dependencies that lack wheels yet.

## Configuration and API key

Secrets stay on your machine only (never commit them).

1. At the **repository root** (next to `pyproject.toml`), copy the env template:

   ```bash
   cp .env.example .env
   ```

2. Edit **`.env`** and set your OpenAI key (and any overrides):

   ```bash
   OPENAI_API_KEY=sk-...your-key-here...
   ```

   The app reads this via [python-dotenv](https://github.com/theskumar/python-dotenv) and [pydantic-settings](../src/book_recommendation/core/settings.py). Optional variables are documented in [`.env.example`](../.env.example).

3. **`.env` is gitignored** — it will not appear in version control and you should not add it manually to Git.

## Real catalog (production data)

The API expects **`books_with_emotions.csv`** and **`tagged_description.txt`** matching the [catalog file contract](#catalog-file-contract) above. They are normally **gitignored**; you build or copy them locally.

**Fast path (no Jupyter):** reproduce the notebook pipeline from the command line (downloads the Kaggle dataset, runs category + emotion models on CPU; expect tens of minutes for the full ~5k-title catalog):

```bash
pip install -e ".[catalog-build]"
python scripts/build_catalog.py
```

Use `python scripts/build_catalog.py --max-books 50` for a quick sanity check. `--dataset-root` points at an existing Kaggle extract (folder containing `books.csv`) if you already downloaded it. The `catalog-build` extra pins compatible **`numpy<2`** / **`transformers<4.46`** so current PyPI **torch 2.2.x** wheels work on many machines; for other stacks, use a separate virtualenv for catalog builds.

**Notebook path** (outputs go to the **repository root**, next to `pyproject.toml`):

1. Install notebook tooling if needed, e.g. from [`requirements-notebooks-full.txt`](../requirements-notebooks-full.txt), and `pip install kagglehub` (or use the Kaggle UI) to access the source dataset in [`notebooks/data-exploration.ipynb`](../notebooks/data-exploration.ipynb) (`dylanjcastillo/7k-books-with-metadata`).
2. Run [`notebooks/data-exploration.ipynb`](../notebooks/data-exploration.ipynb), then [`notebooks/sentiment-analysis.ipynb`](../notebooks/sentiment-analysis.ipynb) — writes **`books_with_emotions.csv`** to the parent directory (repo root).
3. Run [`notebooks/vector-search.ipynb`](../notebooks/vector-search.ipynb) — writes **`tagged_description.txt`** to the repo root (paths like `../tagged_description.txt` assume the notebook cwd is `notebooks/`).
4. Confirm:

   ```bash
   ls -la books_with_emotions.csv tagged_description.txt
   ```

If you store files elsewhere, set **`BOOKS_CSV_PATH`** / **`TAGGED_DESCRIPTION_PATH`** in `.env` to **absolute paths**.

Relative paths in `.env` are resolved from the **repository root** (the folder containing `pyproject.toml`), not from your shell’s current directory — so you can start Uvicorn from any cwd once files are in place.

## Run the API

1. Create a virtual environment and install:

   ```bash
   pip install -e ".[dev]"
   ```

2. Ensure the **catalog** files exist (see **Real catalog** above). At startup the API **validates** paths, CSV columns, and the tagged file’s first ISBN line. If validation fails, check the log message and file locations.

3. Start Uvicorn:

   ```bash
   uvicorn book_recommendation.main:create_app --factory --reload
   ```

   Default URL: [http://127.0.0.1:8000](http://127.0.0.1:8000) (see `/docs` in dev).

## HTTP surface

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness (process is up) |
| GET | `/ready` | Readiness: **200** with `{"status":"ready"}` when the recommendation service and catalog are loaded; **503** otherwise |
| POST | `/v1/recommendations` | Body: `query` (required), `category` (default `All`), `tone` (enum, default `All`), `limit` (1–50, default 16) |

## Security and deployment

- `ENVIRONMENT=prod` disables `/docs`, `/redoc`, and `/openapi.json`.
- `CORS_ORIGINS` is a comma-separated allowlist (e.g. `https://app.example.com`).
- `TRUSTED_HOSTS` is optional; when set, enables `TrustedHostMiddleware` for those hostnames.
- Terminate TLS at your reverse proxy or platform; do not expose internal paths or stack traces to clients (generic `500` body).

## Notebooks

Notebooks live under [`notebooks/`](../notebooks/). They read/write CSV and `tagged_description.txt` in the **parent directory** (repo root), e.g. `../books_cleaned.csv`. Run Jupyter with working directory `notebooks/` or open cells accordingly.

## Testing

```bash
pip install -e ".[dev]"
pytest
```

**Offline** tests cover FastAPI behavior and catalog validation **without** calling OpenAI.

**Live** tests (marker `live`) use the small fixture under [`tests/fixtures/live_minimal/`](../tests/fixtures/live_minimal/) with your real API key. They run only when you opt in:

```bash
export RUN_LIVE=1
export OPENAI_API_KEY=sk-...
pytest -m live
```

Without `RUN_LIVE=1`, live tests are skipped.

### CI

The repository includes a GitHub Actions workflow. The default job runs `pytest` on Python 3.11 and 3.12. An additional job runs `pytest -m live` **only if** the repository defines a `OPENAI_API_KEY` secret.

For a lighter test run without installing the full LangChain/Chroma stack (unit tests mock the service):

```bash
pip install pytest httpx fastapi "uvicorn[standard]" pydantic pydantic-settings pandas numpy python-dotenv typing-extensions
pip install -e . --no-deps
pytest
```

## Optional: full notebook environment

Install everything from [`requirements-notebooks-full.txt`](../requirements-notebooks-full.txt) if you need the historical flat pin set for notebooks or older tooling. The API itself is defined in [`pyproject.toml`](../pyproject.toml).
