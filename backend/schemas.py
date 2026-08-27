from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    top_k: int = Field(default=5, ge=1, le=10)


class SourceReview(BaseModel):
    review_id: str
    text: str
    source: str | None = None
    rating: float | None = None


class QueryResponse(BaseModel):
    answer: str
    evidence_count: int
    sources: list[SourceReview]


class StatsResponse(BaseModel):
    total_reviews: int
    sources: dict[str, int]
