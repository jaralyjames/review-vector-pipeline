from __future__ import annotations

from collections import defaultdict
from pathlib import Path


DISPLAYED_SOURCES = ("play_store", "app_store", "reddit")


def source_statistics(
    persist_directory: Path,
    collection_name: str,
) -> dict[str, object]:
    import chromadb

    if not persist_directory.exists():
        raise FileNotFoundError(
            f"ChromaDB directory not found: {persist_directory}. Run review-ingest first."
        )

    client = chromadb.PersistentClient(path=str(persist_directory))
    try:
        collection = client.get_collection(collection_name)
    except Exception as exc:
        raise ValueError(f"Chroma collection '{collection_name}' was not found.") from exc

    records = collection.get(include=["metadatas"])
    review_ids_by_source: dict[str, set[str]] = defaultdict(set)

    for record_id, metadata in zip(records["ids"], records["metadatas"] or []):
        metadata = metadata or {}
        source = str(metadata.get("source") or "unknown").strip().lower()
        review_id = str(metadata.get("review_id") or record_id)
        review_ids_by_source[source].add(review_id)

    sources = {
        source: len(review_ids_by_source.get(source, set()))
        for source in DISPLAYED_SOURCES
    }
    for source, review_ids in sorted(review_ids_by_source.items()):
        if source not in sources:
            sources[source] = len(review_ids)

    unique_review_ids = set().union(*review_ids_by_source.values()) if review_ids_by_source else set()
    return {"total_reviews": len(unique_review_ids), "sources": sources}
