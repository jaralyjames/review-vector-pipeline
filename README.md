# Review Vector Pipeline

Loads cleaned reviews from CSV, splits long reviews into overlapping token chunks,
creates local embeddings with Sentence Transformers, and stores them in a
persistent ChromaDB collection.

## Project structure

```text
review-vector-pipeline/
├── data/
│   └── cleaned_reviews.sample.csv
├── src/review_vector_pipeline/
│   ├── __init__.py
│   ├── cli.py
│   ├── config.py
│   └── ingest.py
├── tests/
│   └── test_ingest.py
├── .gitignore
├── pyproject.toml
└── requirements.txt
```

## Setup (Windows PowerShell)

```powershell
cd review-vector-pipeline
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

macOS/Linux activation: `source .venv/bin/activate`.

The first run downloads the embedding model. Later runs use the local cache.

## CSV format

The CSV must contain a text column. By default the program checks, in order:
`clean_text`, `text`, `review_text`, `review`, and `content`. Override this with
`--text-column`. If a `metadata_json` column exists, its JSON object is unpacked
into filterable Chroma metadata fields.

All other columns are stored as Chroma metadata when they have a usable scalar
value. A stable `review_id` column is recommended. If it is absent, a content
hash is used.

## Ingest reviews

```powershell
review-ingest data/cleaned_reviews.sample.csv
```

For the included full dataset:

```powershell
review-ingest data/clean_reviews.csv --collection cleaned_reviews
```

Example with explicit options:

```powershell
review-ingest data/cleaned_reviews.csv `
  --text-column review_text `
  --id-column review_id `
  --collection reviews `
  --persist-directory chroma_db `
  --chunk-size 256 `
  --chunk-overlap 40 `
  --batch-size 64
```

You can also run it without the installed command:

```powershell
python -m review_vector_pipeline.cli data/cleaned_reviews.csv
```

Useful options:

- `--model`: Sentence Transformers model name (default: `all-MiniLM-L6-v2`)
- `--device`: `cpu`, `cuda`, or `mps`
- `--reset`: delete and recreate the selected collection before ingestion
- `--limit`: ingest only the first N valid reviews for a quick test

The CLI prints a JSON summary containing review, chunk, skipped-row, and
collection counts.

## Verify with a similarity search

```python
import chromadb
from sentence_transformers import SentenceTransformer

client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection("reviews")
model = SentenceTransformer("all-MiniLM-L6-v2")

query = "Users complain about ads interrupting music"
embedding = model.encode([query], normalize_embeddings=True).tolist()
results = collection.query(query_embeddings=embedding, n_results=5)

for document, metadata, distance in zip(
    results["documents"][0], results["metadatas"][0], results["distances"][0]
):
    print(distance, metadata.get("review_id"), document)
```

## Ask questions with Groq

After ingestion, reinstall the local project once so the new query command and
Groq dependency are available:

```powershell
python -m pip install -e .
```

Create a Groq API key, then set it for the current PowerShell session:

```powershell
$env:GROQ_API_KEY="paste-your-key-here"
```

Ask a question:

```powershell
review-query "What problems do users report about delivery?" `
  --collection cleaned_reviews `
  --persist-directory chroma_db
```

Equivalent module command:

```powershell
python -m review_vector_pipeline.query_cli `
  "What problems do users report about delivery?" `
  --collection cleaned_reviews `
  --persist-directory chroma_db
```

The retrieval layer embeds the question with the same Sentence Transformer used
during ingestion, retrieves five unique relevant reviews from ChromaDB, and
sends only those reviews plus the question to Groq. The answer cites evidence as
`[Review 1]`, `[Review 2]`, and so on, followed by the retrieved review details.

Optional metadata filters:

```powershell
review-query "What are the main complaints?" --source play_store --max-rating 2
```

Use `--llm-model` to select another model enabled in your Groq account.

## Run tests

```powershell
pip install pytest
pytest
```

## Run the web application

The web application has a FastAPI backend and a minimal Next.js frontend. The
backend uses the existing retrieval pipeline and ChromaDB collection.

### Backend setup

Copy `.env.example` to `.env`, then set your real `GROQ_API_KEY`. Confirm that
`CHROMA_COLLECTION` matches the collection created by `review-ingest`.

```powershell
Copy-Item .env.example .env
python -m pip install -e .
uvicorn backend.main:app --reload --port 8000
```

Verify the backend at `http://127.0.0.1:8000/api/health` and view its API docs at
`http://127.0.0.1:8000/docs`.

### Frontend setup

Open another PowerShell window:

```powershell
cd frontend
Copy-Item .env.local.example .env.local
npm install
npm run dev
```

Open `http://localhost:3000`. Clicking an example question fills the chat box.
After an answer is generated, the collapsed **Sources** link reveals the five
reviews used as evidence.

### Production configuration

- Set `NEXT_PUBLIC_API_URL` to the public FastAPI URL before building Next.js.
- Add that frontend origin to `FRONTEND_ORIGINS` on the backend.
- Keep `GROQ_API_KEY` only on the backend.
- Keep the `chroma_db` directory on persistent storage.

## Notes

- Chunk IDs are deterministic hashes, and ingestion uses `upsert`.
- Empty or whitespace-only text rows are skipped.
- Embeddings are normalized, making Chroma's cosine distance appropriate.
- Keep `chroma_db/` out of Git; it can be rebuilt from the source CSV.
