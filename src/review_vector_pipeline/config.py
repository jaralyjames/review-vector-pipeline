from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class IngestConfig:
    csv_path: Path
    persist_directory: Path = Path("chroma_db")
    collection_name: str = "reviews"
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    text_column: str | None = None
    id_column: str = "review_id"
    chunk_size: int = 256
    chunk_overlap: int = 40
    batch_size: int = 64
    device: str | None = None
    reset: bool = False
    limit: int | None = None

    def validate(self) -> None:
        if not self.csv_path.is_file():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be at least 1")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if self.limit is not None and self.limit < 1:
            raise ValueError("limit must be at least 1")

