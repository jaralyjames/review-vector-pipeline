from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.analytics import source_statistics
from backend.schemas import QueryRequest, QueryResponse, SourceReview, StatsResponse
from review_vector_pipeline.retrieval import retrieve_and_answer


BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

CHROMA_DIRECTORY = Path(
    os.getenv("CHROMA_DIRECTORY", str(BASE_DIR / "chroma_db"))
).resolve()
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "cleaned_reviews")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
FRONTEND_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "FRONTEND_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000,https://myntra-review-analyser.vercel.app "
    ).split(",")
    if origin.strip()
]

app = FastAPI(title="Myntra Review Analyser API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.get("/api/stats", response_model=StatsResponse)
def stats() -> StatsResponse:
    try:
        result = source_statistics(CHROMA_DIRECTORY, COLLECTION_NAME)
        return StatsResponse(**result)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/query", response_model=QueryResponse)
def query_reviews(request: QueryRequest) -> QueryResponse:
    try:
        result = retrieve_and_answer(
            question=request.question,
            persist_directory=CHROMA_DIRECTORY,
            collection_name=COLLECTION_NAME,
            llm_model=GROQ_MODEL,
            top_k=request.top_k,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Unable to generate the review analysis.") from exc

    sources = [
        SourceReview(
            review_id=review.review_id,
            text=review.text,
            source=(str(review.metadata["source"]) if review.metadata.get("source") else None),
            rating=(float(review.metadata["rating"]) if review.metadata.get("rating") is not None else None),
        )
        for review in result.reviews
    ]
    return QueryResponse(answer=result.answer, evidence_count=len(sources), sources=sources)
