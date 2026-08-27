from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import IngestConfig
from .ingest import ingest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Chunk cleaned CSV reviews, embed them, and store them in ChromaDB."
    )
    parser.add_argument("csv_path", type=Path, help="Path to the cleaned reviews CSV")
    parser.add_argument("--persist-directory", type=Path, default=Path("chroma_db"))
    parser.add_argument("--collection", default="reviews", dest="collection_name")
    parser.add_argument(
        "--model", default="sentence-transformers/all-MiniLM-L6-v2", dest="model_name"
    )
    parser.add_argument("--text-column")
    parser.add_argument("--id-column", default="review_id")
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--chunk-overlap", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"))
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--limit", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = IngestConfig(**vars(args))
    try:
        summary = ingest(config)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
    print(json.dumps(summary.to_dict(), indent=2))


if __name__ == "__main__":
    main()

