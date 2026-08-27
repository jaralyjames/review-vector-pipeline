from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from .config import IngestConfig

TEXT_COLUMN_CANDIDATES = ("clean_text", "text", "review_text", "review", "content")


@dataclass(frozen=True)
class Review:
    review_id: str
    text: str
    metadata: dict[str, str | int | float | bool]


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    metadata: dict[str, str | int | float | bool]


@dataclass(frozen=True)
class IngestSummary:
    reviews_read: int
    reviews_ingested: int
    rows_skipped: int
    chunks_upserted: int
    collection_count: int
    collection_name: str
    persist_directory: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def resolve_text_column(fieldnames: Sequence[str], requested: str | None) -> str:
    if requested:
        if requested not in fieldnames:
            raise ValueError(
                f"Text column '{requested}' is missing. Available columns: {list(fieldnames)}"
            )
        return requested
    for candidate in TEXT_COLUMN_CANDIDATES:
        if candidate in fieldnames:
            return candidate
    raise ValueError(
        "No review text column found. Add one of "
        f"{list(TEXT_COLUMN_CANDIDATES)} or pass --text-column. "
        f"Available columns: {list(fieldnames)}"
    )


def metadata_value(value: Any) -> str | int | float | bool | None:
    """Convert CSV values to scalar values accepted by Chroma metadata."""
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    cleaned = " ".join(str(value).split())
    return cleaned or None


def parse_metadata_json(raw_value: str | None, row_number: int) -> dict[str, Any]:
    if not raw_value or not raw_value.strip():
        return {}
    try:
        parsed = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid metadata_json at CSV row {row_number}: {exc.msg}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"metadata_json at CSV row {row_number} must be a JSON object")
    return parsed


def read_reviews(
    csv_path: Path,
    text_column: str | None = None,
    id_column: str = "review_id",
    limit: int | None = None,
) -> tuple[list[Review], int, int]:
    reviews: list[Review] = []
    skipped = 0
    rows_read = 0

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("CSV has no header row")
        selected_text_column = resolve_text_column(reader.fieldnames, text_column)

        for row_number, row in enumerate(reader, start=2):
            rows_read += 1
            text = " ".join((row.get(selected_text_column) or "").split())
            if not text:
                skipped += 1
                continue

            supplied_id = " ".join((row.get(id_column) or "").split())
            review_id = supplied_id or stable_hash(text)[:24]
            metadata: dict[str, str | int | float | bool] = {}
            json_metadata = parse_metadata_json(row.get("metadata_json"), row_number)
            for key, raw_value in json_metadata.items():
                converted = metadata_value(raw_value)
                if converted is not None:
                    metadata[str(key)] = converted
            for key, raw_value in row.items():
                if key in (selected_text_column, "metadata_json"):
                    continue
                converted = metadata_value(raw_value)
                if converted is not None:
                    metadata[key] = converted
            metadata[id_column] = review_id
            metadata["source_row"] = row_number
            reviews.append(Review(review_id=review_id, text=text, metadata=metadata))
            if limit is not None and len(reviews) >= limit:
                break

    return reviews, rows_read, skipped


def chunk_review(review: Review, tokenizer: Any, size: int, overlap: int) -> list[Chunk]:
    token_ids: list[int] = tokenizer.encode(review.text, add_special_tokens=False)
    if not token_ids:
        return []

    step = size - overlap
    pieces: list[Chunk] = []
    for index, start in enumerate(range(0, len(token_ids), step)):
        window = token_ids[start : start + size]
        if not window:
            break
        chunk_text = tokenizer.decode(window, skip_special_tokens=True).strip()
        if not chunk_text:
            continue
        chunk_id = stable_hash(f"{review.review_id}:{index}:{chunk_text}")
        metadata = {
            **review.metadata,
            "chunk_index": index,
            "token_start": start,
            "token_count": len(window),
        }
        pieces.append(Chunk(chunk_id=chunk_id, text=chunk_text, metadata=metadata))
        if start + size >= len(token_ids):
            break
    return pieces


def batched(items: Sequence[Chunk], size: int) -> Iterator[Sequence[Chunk]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def ingest(config: IngestConfig) -> IngestSummary:
    config.validate()

    import chromadb
    from sentence_transformers import SentenceTransformer

    reviews, rows_read, rows_skipped = read_reviews(
        config.csv_path, config.text_column, config.id_column, config.limit
    )
    if not reviews:
        raise ValueError("No usable review text was found in the CSV")

    model = SentenceTransformer(config.model_name, device=config.device)
    chunks = [
        chunk
        for review in reviews
        for chunk in chunk_review(
            review, model.tokenizer, config.chunk_size, config.chunk_overlap
        )
    ]
    if not chunks:
        raise ValueError("The tokenizer produced no usable chunks")

    config.persist_directory.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(config.persist_directory))
    if config.reset:
        try:
            client.delete_collection(config.collection_name)
        except Exception as exc:
            if "does not exist" not in str(exc).lower() and "not found" not in str(exc).lower():
                raise

    collection = client.get_or_create_collection(
        name=config.collection_name,
        metadata={"hnsw:space": "cosine", "embedding_model": config.model_name},
    )

    for batch in batched(chunks, config.batch_size):
        documents = [chunk.text for chunk in batch]
        embeddings = model.encode(
            documents,
            batch_size=config.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        ).tolist()
        collection.upsert(
            ids=[chunk.chunk_id for chunk in batch],
            documents=documents,
            metadatas=[chunk.metadata for chunk in batch],
            embeddings=embeddings,
        )

    return IngestSummary(
        reviews_read=rows_read,
        reviews_ingested=len(reviews),
        rows_skipped=rows_skipped,
        chunks_upserted=len(chunks),
        collection_count=collection.count(),
        collection_name=config.collection_name,
        persist_directory=str(config.persist_directory.resolve()),
    )
