from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class RetrievedReview:
    rank: int
    review_id: str
    text: str
    distance: float | None
    metadata: dict[str, Any]


@dataclass(frozen=True)
class AnswerResult:
    answer: str
    reviews: list[RetrievedReview]


SYSTEM_PROMPT = """You are a customer review analysis assistant.

Answer the user's question using only the retrieved customer reviews provided.

Provide one consolidated and concise brief that combines the relevant findings
across the reviews.

Requirements:
- Do not list individual reviews.
- Do not quote individual reviews.
- Do not mention review numbers or review IDs.
- Do not add citations such as [Review 1].
- Do not explain the retrieval process.
- Do not invent facts, statistics, frequencies, causes, or product details.
- If the retrieved reviews do not provide sufficient evidence, clearly state that.
- Write the answer as a clear, natural summary for a product manager.
"""


def retrieve_reviews(
    question: str,
    persist_directory: Path,
    collection_name: str,
    embedding_model: str,
    top_k: int = 5,
    device: str | None = None,
    where: dict[str, Any] | None = None,
) -> list[RetrievedReview]:
    if not question.strip():
        raise ValueError("Question cannot be empty")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if not persist_directory.exists():
        raise FileNotFoundError(
            f"ChromaDB directory not found: {persist_directory}. Run review-ingest first."
        )

    import chromadb
    from sentence_transformers import SentenceTransformer

    client = chromadb.PersistentClient(path=str(persist_directory))
    try:
        collection = client.get_collection(collection_name)
    except Exception as exc:
        raise ValueError(
            f"Chroma collection '{collection_name}' was not found. Run review-ingest first."
        ) from exc

    model = SentenceTransformer(embedding_model, device=device)
    query_embedding = model.encode(
        [question.strip()], normalize_embeddings=True, show_progress_bar=False
    ).tolist()
    collection_count = collection.count()
    if collection_count == 0:
        return []
    query_args: dict[str, Any] = {
        "query_embeddings": query_embedding,
        # Fetch extra chunks so duplicate chunks from one review do not crowd out results.
        "n_results": min(top_k * 3, collection_count),
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        query_args["where"] = where
    result = collection.query(**query_args)

    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]
    ids = result.get("ids", [[]])[0]
    reviews: list[RetrievedReview] = []
    seen_review_ids: set[str] = set()
    for chunk_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
        metadata = metadata or {}
        review_id = str(metadata.get("review_id") or chunk_id)
        if review_id in seen_review_ids:
            continue
        seen_review_ids.add(review_id)
        reviews.append(RetrievedReview(
            rank=len(reviews) + 1,
            review_id=review_id,
            text=document,
            distance=float(distance) if distance is not None else None,
            metadata=metadata,
        ))
    return reviews[:top_k]


def build_llm_input(question: str, reviews: list[RetrievedReview]) -> str:
    evidence = []
    for review in reviews:
        source = review.metadata.get("source", "unknown")
        rating = review.metadata.get("rating", "unknown")
        evidence.append(
            f"[Review {review.rank}] ID: {review.review_id}\n"
            f"Source: {source}; Rating: {rating}\nText: {review.text}"
        )
    joined = "\n\n".join(evidence) or "No reviews were retrieved."
    return f"User question:\n{question.strip()}\n\nRetrieved reviews:\n{joined}"


def answer_with_groq(
    question: str,
    reviews: list[RetrievedReview],
    model: str = "openai/gpt-oss-20b",
) -> str:
    if not os.getenv("GROQ_API_KEY"):
        raise ValueError("GROQ_API_KEY is not set. Add it as an environment variable before querying.")
    if not reviews:
        return "I could not find relevant reviews to answer that question."

    from groq import Groq

    completion = Groq().chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_llm_input(question, reviews)},
        ],
    )
    answer = completion.choices[0].message.content
    if not answer:
        raise RuntimeError("Groq returned an empty answer")
    return answer.strip()


def retrieve_and_answer(
    question: str,
    persist_directory: Path = Path("chroma_db"),
    collection_name: str = "cleaned_reviews",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    llm_model: str = "openai/gpt-oss-20b",
    top_k: int = 5,
    device: str | None = None,
    where: dict[str, Any] | None = None,
) -> AnswerResult:
    reviews = retrieve_reviews(
        question, persist_directory, collection_name, embedding_model, top_k, device, where
    )
    return AnswerResult(
        answer=answer_with_groq(question, reviews, model=llm_model), reviews=reviews
    )
