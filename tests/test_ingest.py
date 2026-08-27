import csv
from pathlib import Path

import pytest

from review_vector_pipeline.config import IngestConfig
from review_vector_pipeline.ingest import Review, chunk_review, read_reviews


class WordTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(range(len(text.split())))

    def decode(self, token_ids, skip_special_tokens=True):
        return " ".join(f"word{token_id}" for token_id in token_ids)


def test_chunking_uses_overlap_and_stable_ids():
    review = Review("r1", "one two three four five six", {"rating": "5"})
    chunks = chunk_review(review, WordTokenizer(), size=4, overlap=1)

    assert [chunk.metadata["token_start"] for chunk in chunks] == [0, 3]
    assert [chunk.metadata["token_count"] for chunk in chunks] == [4, 3]
    assert chunks == chunk_review(review, WordTokenizer(), size=4, overlap=1)


def test_read_reviews_accepts_review_text_and_skips_empty(tmp_path: Path):
    path = tmp_path / "reviews.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["review_id", "review_text", "rating"])
        writer.writeheader()
        writer.writerow({"review_id": "a", "review_text": " Great app! ", "rating": "5"})
        writer.writerow({"review_id": "b", "review_text": "   ", "rating": "1"})

    reviews, rows_read, skipped = read_reviews(path)

    assert rows_read == 2
    assert skipped == 1
    assert reviews[0].text == "Great app!"
    assert reviews[0].metadata["review_id"] == "a"


def test_read_reviews_unpacks_metadata_json(tmp_path: Path):
    path = tmp_path / "reviews.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["review_id", "clean_text", "metadata_json"]
        )
        writer.writeheader()
        writer.writerow(
            {
                "review_id": "x1",
                "clean_text": "Useful app",
                "metadata_json": '{"source":"play_store","rating":4.0}',
            }
        )

    reviews, _, _ = read_reviews(path)
    assert reviews[0].metadata["source"] == "play_store"
    assert reviews[0].metadata["rating"] == 4.0
    assert "metadata_json" not in reviews[0].metadata


def test_config_rejects_invalid_overlap(tmp_path: Path):
    path = tmp_path / "reviews.csv"
    path.write_text("review_text\nhello\n", encoding="utf-8")
    with pytest.raises(ValueError, match="overlap"):
        IngestConfig(csv_path=path, chunk_size=10, chunk_overlap=10).validate()
