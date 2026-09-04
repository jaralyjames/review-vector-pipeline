from __future__ import annotations

import argparse
import json
from pathlib import Path

from .retrieval import retrieve_and_answer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Retrieve ChromaDB reviews and answer with Groq.")
    parser.add_argument("question", help="Question to ask about the reviews")
    parser.add_argument("--persist-directory", type=Path, default=Path("chroma_db"))
    parser.add_argument("--collection", default="cleaned_reviews", dest="collection_name")
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--llm-model", default="llama-3.3-70b-versatile")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"))
    parser.add_argument("--source", help="Optional exact source metadata filter")
    parser.add_argument("--max-rating", type=float, help="Optional maximum rating filter")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    filters = []
    if args.source:
        filters.append({"source": args.source})
    if args.max_rating is not None:
        filters.append({"rating": {"$lte": args.max_rating}})
    where = filters[0] if len(filters) == 1 else ({"$and": filters} if filters else None)
    try:
        result = retrieve_and_answer(
            args.question, args.persist_directory, args.collection_name,
            args.embedding_model, args.llm_model, args.top_k, args.device, where
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"Error: {exc}") from exc

    print("\nANSWER\n------")
    print(result.answer)
    print("\nRETRIEVED REVIEWS\n-----------------")
    for review in result.reviews:
        print(json.dumps({
            "rank": review.rank,
            "review_id": review.review_id,
            "source": review.metadata.get("source"),
            "rating": review.metadata.get("rating"),
            "distance": round(review.distance, 4) if review.distance is not None else None,
            "text": review.text,
        }, ensure_ascii=False))


if __name__ == "__main__":
    main()
