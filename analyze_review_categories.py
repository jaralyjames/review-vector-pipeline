"""Classify ChromaDB reviews, summarize sentiment, and rank wishlist opportunities.

Every unique review receives exactly one primary category and one sentiment.
Classification is cached back into every Chroma chunk, so later runs process only
new reviews. Percentages are review evidence, not measured conversion rates.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    import chromadb
except ImportError:
    chromadb = None

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

try:
    from groq import Groq
except ImportError:
    Groq = None

if load_dotenv:
    load_dotenv()


ANALYSIS_VERSION = 1
QUESTION_ANALYSIS_VERSION = 2
CATEGORIES = (
    "product_quality_and_accuracy",
    "size_fit_and_availability",
    "pricing_discounts_and_value",
    "wishlist_and_cart_reliability",
    "delivery_returns_and_refunds",
    "support_payment_and_app_experience",
)
SENTIMENTS = ("positive", "negative", "neutral")
WISHLIST_BARRIERS = (
    "wishlist_reliability",
    "stock_and_size_availability",
    "price_and_discount_transparency",
    "price_drop_communication",
    "product_confidence",
    "wishlist_organization",
    "cart_and_checkout_friction",
    "decision_support",
    "cross_device_continuity",
    "purchase_reminders",
    "not_wishlist_related",
)

CATEGORY_LABELS = {
    "product_quality_and_accuracy": "Product quality & accuracy",
    "size_fit_and_availability": "Size, fit & availability",
    "pricing_discounts_and_value": "Pricing, discounts & value",
    "wishlist_and_cart_reliability": "Wishlist & cart reliability",
    "delivery_returns_and_refunds": "Delivery, returns & refunds",
    "support_payment_and_app_experience": "Support, payment & app experience",
}

INTERVENTIONS = {
    "wishlist_reliability": "Make saved-item state reliable across sessions.",
    "stock_and_size_availability": "Offer preferred-size and restock alerts.",
    "price_and_discount_transparency": "Show final eligible price and price history.",
    "price_drop_communication": "Send relevant, timely price-drop alerts.",
    "product_confidence": "Improve size, quality, authenticity and review information.",
    "wishlist_organization": "Add wishlist search, filters and collections.",
    "cart_and_checkout_friction": "Simplify move-to-cart and checkout recovery.",
    "decision_support": "Add comparison and concise product decision support.",
    "cross_device_continuity": "Synchronize saved items across web, app and devices.",
    "purchase_reminders": "Use contextual reminders tied to price or availability.",
}

CLASSIFY_PROMPT = f"""Classify online-shopping reviews. Return JSON only as
{{"results": [...]}} with exactly one result per input row_key. Each result needs:
row_key, primary_category, sentiment, confidence, wishlist_relevant,
wishlist_barrier, purchase_intent, purchase_outcome, severity.

primary_category must be exactly one of: {', '.join(CATEGORIES)}.
sentiment: positive, negative, or neutral. Judge the review's overall sentiment.
confidence: number from 0 to 1.
wishlist_relevant: boolean. True only when the review discusses saved/wishlisted/
favourited items, cart movement from saved items, or a clear barrier preventing a
saved item from purchase.
wishlist_barrier must be one of: {', '.join(WISHLIST_BARRIERS)}.
purchase_intent: low, medium, high, or unknown.
purchase_outcome: purchased, postponed, abandoned, or unknown.
severity: low, medium, or high.

Category boundaries:
- product_quality_and_accuracy: quality, authenticity, damage, description mismatch.
- size_fit_and_availability: sizing, fit, size chart, stock or unavailable sizes.
- pricing_discounts_and_value: price, coupon, discount, fees or value.
- wishlist_and_cart_reliability: saving, wishlist, cart, persistence or checkout transfer.
- delivery_returns_and_refunds: shipping, packaging, cancellation, return or refund.
- support_payment_and_app_experience: support, seller conduct, payment, login, speed,
  crash, navigation or other app/site experience.
Choose the single main issue. Use not_wishlist_related when wishlist_relevant is false.
Do not infer details absent from the review."""

SUMMARY_PROMPT = """Summarize the supplied online-shopping reviews in no more than
55 words. Focus on the stated dominant sentiment and its main recurring reasons.
Mention one customer expectation when supported. Do not invent numbers, causes,
features or facts. Return JSON only: {"summary": "..."}."""

QUESTIONS = (
    "Why do users add fashion products to their wishlist?",
    "What prevents wishlisted products from eventually being purchased?",
    "What uncertainties remain after users have identified a product they like?",
    "What causes users to postpone a purchase?",
    "How do users compare multiple shortlisted products?",
    "What information do users seek outside Myntra/AJIO before purchasing?",
    "What role do fit, size, styling, price, reviews, occasion and social validation play?",
    "When do users use the wishlist as genuine purchase intent versus simply as a bookmarking mechanism?",
    "How do these behaviors differ across user segments?",
    "What unmet needs emerge consistently across user conversations?",
)

QUESTION_PROMPT = """Answer the supplied ecommerce research questions using only
the supplied customer reviews. The evidence has already been filtered to remove
duplicates and routine, low-information praise. Treat repeated themes as one signal
rather than allowing repeated wording to dominate the answer.
Return JSON only as {"results": [...]}. Each result must include row_key, question,
answer, and sample_review_ids. The answer must be a concise 45-80 word synthesis.
sample_review_ids must contain exactly two different review IDs that directly support
the answer. Prioritize specific needs, barriers, uncertainties and purchase behavior.
Do not invent statistics, user segments, motives or facts. When evidence is
insufficient, explicitly state that limitation and select the two closest relevant
reviews. Return one result for every question."""


@dataclass
class Review:
    review_id: str
    text: str
    source: str
    chunks: list[tuple[str, dict[str, Any]]]
    primary_category: str = ""
    sentiment: str = ""
    confidence: float = 0.0
    wishlist_relevant: bool = False
    wishlist_barrier: str = "not_wishlist_related"
    purchase_intent: str = "unknown"
    purchase_outcome: str = "unknown"
    severity: str = "low"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--persist-directory", type=Path,
                        default=Path(os.getenv("CHROMA_DIRECTORY", "chroma_db")))
    parser.add_argument("--collection", default=os.getenv("CHROMA_COLLECTION", "cleaned_reviews"))
    parser.add_argument("--model", default=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"))
    parser.add_argument("--read-batch-size", type=int, default=500)
    parser.add_argument("--classification-batch-size", type=int, default=10)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--summary-sample-size", type=int, default=20)
    parser.add_argument("--samples-per-opportunity", type=int, default=2)
    parser.add_argument("--question-batch-size", type=int, default=5)
    parser.add_argument("--question-evidence-size", type=int, default=36)
    parser.add_argument("--regenerate-question-answers", action="store_true")
    parser.add_argument("--overwrite-analysis", action="store_true")
    parser.add_argument("--output-directory", type=Path, default=Path("data"))
    return parser.parse_args()


def load_reviews(collection: Any, batch_size: int) -> list[Review]:
    grouped: dict[str, dict[str, Any]] = {}
    for offset in range(0, collection.count(), batch_size):
        page = collection.get(limit=batch_size, offset=offset,
                              include=["documents", "metadatas"])
        for chunk_id, document, metadata in zip(
            page.get("ids") or [], page.get("documents") or [],
            page.get("metadatas") or [], strict=True,
        ):
            metadata = dict(metadata or {})
            review_id = str(metadata.get("review_id") or chunk_id)
            item = grouped.setdefault(review_id, {"parts": [], "chunks": [], "source": "unknown"})
            item["parts"].append((int(metadata.get("chunk_index", 0)), str(document or "").strip()))
            item["chunks"].append((str(chunk_id), metadata))
            if metadata.get("source"):
                item["source"] = str(metadata["source"]).lower()

    reviews: list[Review] = []
    for review_id, item in sorted(grouped.items()):
        text = " ".join(part for _, part in sorted(item["parts"]) if part).strip()
        if not text:
            continue
        review = Review(review_id, text, item["source"], item["chunks"])
        metadata = next((m for _, m in item["chunks"]
                         if m.get("review_analysis_version") == ANALYSIS_VERSION), None)
        if metadata:
            apply_analysis(review, metadata)
        reviews.append(review)
    return reviews


def apply_analysis(review: Review, result: dict[str, Any]) -> None:
    review.primary_category = str(result["primary_category"])
    review.sentiment = str(result["sentiment"])
    review.confidence = float(result["confidence"])
    review.wishlist_relevant = bool(result["wishlist_relevant"])
    review.wishlist_barrier = str(result["wishlist_barrier"])
    review.purchase_intent = str(result["purchase_intent"])
    review.purchase_outcome = str(result["purchase_outcome"])
    review.severity = str(result["severity"])


def validate_result(value: object, row_key: str) -> dict[str, Any]:
    if not isinstance(value, dict) or str(value.get("row_key")) != row_key:
        raise ValueError(f"Invalid row_key {row_key}")
    result = dict(value)
    allowed = {
        "primary_category": set(CATEGORIES), "sentiment": set(SENTIMENTS),
        "wishlist_barrier": set(WISHLIST_BARRIERS),
        "purchase_intent": {"low", "medium", "high", "unknown"},
        "purchase_outcome": {"purchased", "postponed", "abandoned", "unknown"},
        "severity": {"low", "medium", "high"},
    }
    for field, choices in allowed.items():
        result[field] = str(result.get(field, "")).strip().lower()
        if result[field] not in choices:
            raise ValueError(f"Invalid {field} for row {row_key}")
    if not isinstance(result.get("wishlist_relevant"), bool):
        raise ValueError(f"wishlist_relevant must be boolean for row {row_key}")
    result["confidence"] = float(result.get("confidence"))
    if not 0 <= result["confidence"] <= 1:
        raise ValueError(f"Invalid confidence for row {row_key}")
    if not result["wishlist_relevant"]:
        result["wishlist_barrier"] = "not_wishlist_related"
    elif result["wishlist_barrier"] == "not_wishlist_related":
        raise ValueError(f"Wishlist-related row {row_key} needs a barrier")
    return result


def call_json(client: Any, model: str, system: str, payload: object,
              max_retries: int) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model, temperature=0, response_format={"type": "json_object"},
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
            )
            return json.loads(response.choices[0].message.content)
        except Exception as error:
            last_error = error
            if attempt + 1 < max_retries:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Groq request failed after {max_retries} attempts: {last_error}")


def classify_batch(client: Any, model: str, reviews: list[Review],
                   max_retries: int) -> list[dict[str, Any]]:
    payload = [{"row_key": str(i), "review": review.text}
               for i, review in enumerate(reviews)]
    try:
        parsed = call_json(client, model, CLASSIFY_PROMPT, payload, max_retries)
        values = parsed.get("results")
        if not isinstance(values, list) or len(values) != len(reviews):
            raise ValueError("Model returned the wrong number of results")
        indexed = {str(value.get("row_key")): value for value in values if isinstance(value, dict)}
        return [validate_result(indexed[str(i)], str(i)) for i in range(len(reviews))]
    except Exception:
        if len(reviews) == 1:
            raise
        middle = len(reviews) // 2
        return (classify_batch(client, model, reviews[:middle], max_retries)
                + classify_batch(client, model, reviews[middle:], max_retries))


def save_analysis(collection: Any, review: Review, result: dict[str, Any]) -> None:
    metadata_result = {key: value for key, value in result.items() if key != "row_key"}
    metadata_result["review_analysis_version"] = ANALYSIS_VERSION
    collection.update(
        ids=[chunk_id for chunk_id, _ in review.chunks],
        metadatas=[{**metadata, **metadata_result} for _, metadata in review.chunks],
    )
    apply_analysis(review, metadata_result)


def classify_reviews(collection: Any, reviews: list[Review], client: Any, model: str,
                     batch_size: int, max_retries: int, overwrite: bool) -> None:
    pending = [review for review in reviews if overwrite or not review.primary_category]
    if not pending:
        print(f"All {len(reviews)} reviews already have current classification.")
        return
    print(f"Classifying {len(pending)} of {len(reviews)} unique reviews...")
    for start in range(0, len(pending), batch_size):
        batch = pending[start:start + batch_size]
        results = classify_batch(client, model, batch, max_retries)
        for review, result in zip(batch, results, strict=True):
            save_analysis(collection, review, result)
        print(f"Classified {min(start + len(batch), len(pending))}/{len(pending)}")


def dominant_sentiment(counts: Counter[str]) -> str:
    highest = max(counts.get(sentiment, 0) for sentiment in SENTIMENTS)
    winners = [sentiment for sentiment in SENTIMENTS if counts.get(sentiment, 0) == highest]
    return winners[0] if len(winners) == 1 else "mixed"


def summarize_category(client: Any, model: str, category: str, sentiment: str,
                       reviews: list[Review], sample_size: int, max_retries: int) -> str:
    candidates = [review for review in reviews if sentiment == "mixed" or review.sentiment == sentiment]
    candidates.sort(key=lambda review: review.confidence, reverse=True)
    # Prefer source diversity, then fill remaining slots by confidence.
    sample: list[Review] = []
    seen_sources: set[str] = set()
    for review in candidates:
        if review.source not in seen_sources and len(sample) < sample_size:
            sample.append(review); seen_sources.add(review.source)
    for review in candidates:
        if review not in sample and len(sample) < sample_size:
            sample.append(review)
    payload = {"category": CATEGORY_LABELS[category], "dominant_sentiment": sentiment,
               "reviews": [{"source": r.source, "text": r.text} for r in sample]}
    parsed = call_json(client, model, SUMMARY_PROMPT, payload, max_retries)
    summary = str(parsed.get("summary") or "").strip()
    if not summary:
        raise ValueError(f"Empty summary for {category}")
    return summary


def build_category_results(reviews: list[Review], client: Any, model: str,
                           sample_size: int, max_retries: int) -> list[dict[str, Any]]:
    results = []
    total = len(reviews)
    for category in CATEGORIES:
        matches = [review for review in reviews if review.primary_category == category]
        counts = Counter(review.sentiment for review in matches)
        dominant = dominant_sentiment(counts) if matches else "no_reviews"
        summary = (summarize_category(client, model, category, dominant, matches,
                                      sample_size, max_retries) if matches else
                   "No reviews were assigned to this category.")
        results.append({
            "category": category, "label": CATEGORY_LABELS[category],
            "review_count": len(matches),
            "review_share_percent": round(len(matches) / total * 100, 1) if total else 0,
            "sentiments": {sentiment: {"count": counts.get(sentiment, 0),
                "percent": round(counts.get(sentiment, 0) / len(matches) * 100, 1) if matches else 0}
                for sentiment in SENTIMENTS},
            "dominant_sentiment": dominant, "summary": summary,
            "source_counts": dict(Counter(review.source for review in matches)),
        })
    return results


def rank_wishlist_opportunities(reviews: list[Review], sample_count: int) -> list[dict[str, Any]]:
    wishlist = [review for review in reviews if review.wishlist_relevant]
    opportunities = []
    for barrier in WISHLIST_BARRIERS[:-1]:
        matches = [review for review in wishlist if review.wishlist_barrier == barrier]
        if not matches:
            continue
        negative_rate = sum(r.sentiment == "negative" for r in matches) / len(matches) * 100
        abandonment_rate = sum(r.purchase_outcome == "abandoned" for r in matches) / len(matches) * 100
        high_intent_rate = sum(r.purchase_intent == "high" for r in matches) / len(matches) * 100
        high_severity_rate = sum(r.severity == "high" for r in matches) / len(matches) * 100
        share = len(matches) / len(wishlist) * 100 if wishlist else 0
        source_score = min(len({r.source for r in matches}) / 3, 1) * 100
        score = round(share * .30 + abandonment_rate * .25 + high_severity_rate * .20
                      + high_intent_rate * .15 + source_score * .10, 1)
        samples = sorted(matches, key=lambda r: (r.sentiment == "negative", r.confidence), reverse=True)[:sample_count]
        opportunities.append({
            "barrier": barrier, "mentions": len(matches),
            "wishlist_review_share_percent": round(share, 1),
            "negative_rate_percent": round(negative_rate, 1),
            "abandonment_rate_percent": round(abandonment_rate, 1),
            "high_intent_rate_percent": round(high_intent_rate, 1),
            "high_severity_rate_percent": round(high_severity_rate, 1),
            "opportunity_score": score, "suggested_improvement": INTERVENTIONS[barrier],
            "samples": [{"review_id": r.review_id, "source": r.source,
                         "sentiment": r.sentiment, "text": r.text} for r in samples],
        })
    opportunities.sort(key=lambda item: (-item["opportunity_score"], -item["mentions"]))
    for rank, opportunity in enumerate(opportunities, 1):
        opportunity["rank"] = rank
    return opportunities


def review_tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def is_routine_positive(review: Review) -> bool:
    """Reject generic praise that carries little research information."""
    if review.sentiment != "positive":
        return False
    tokens = review_tokens(review.text)
    generic_words = {
        "amazing", "app", "awesome", "best", "excellent", "experience", "fashion",
        "good", "great", "like", "love", "lovely", "nice", "perfect", "product",
        "shopping", "super", "very", "wow",
    }
    specific_signals = {
        "cart", "compare", "coupon", "delivery", "discount", "fit", "occasion",
        "price", "refund", "return", "review", "save", "size", "stock", "style",
        "wishlist",
    }
    return len(tokens) < 10 or (tokens <= generic_words and not tokens & specific_signals)


def sufficiently_different(candidate: Review, selected: list[Review]) -> bool:
    candidate_tokens = review_tokens(candidate.text)
    if not candidate_tokens:
        return False
    for existing in selected:
        existing_tokens = review_tokens(existing.text)
        union = candidate_tokens | existing_tokens
        if union and len(candidate_tokens & existing_tokens) / len(union) >= 0.78:
            return False
    return True


def evidence_information_score(review: Review) -> tuple[int, int, float]:
    signals = {
        "because", "but", "cart", "compare", "discount", "fit", "price", "refund",
        "return", "review", "size", "stock", "wish", "wishlist", "would",
    }
    tokens = review_tokens(review.text)
    return (len(tokens & signals), min(len(tokens), 80), review.confidence)


def select_question_evidence(reviews: list[Review], limit: int) -> tuple[str, list[Review]]:
    overall = dominant_sentiment(Counter(review.sentiment for review in reviews))
    dominant_reviews = reviews if overall == "mixed" else [review for review in reviews if review.sentiment == overall]
    eligible = [review for review in dominant_reviews if not is_routine_positive(review)]
    if len(eligible) < 2:
        eligible = [review for review in reviews if not is_routine_positive(review)]
    # Rank specific reviews above generic comments, then remove near-duplicates.
    ranked = sorted(eligible, key=evidence_information_score, reverse=True)
    unique_ranked: list[Review] = []
    for review in ranked:
        if sufficiently_different(review, unique_ranked):
            unique_ranked.append(review)
    selected: list[Review] = []
    seen_pairs: set[tuple[str, str]] = set()
    for review in unique_ranked:
        pair = (review.primary_category, review.source)
        if pair not in seen_pairs and len(selected) < limit:
            selected.append(review)
            seen_pairs.add(pair)
    for review in unique_ranked:
        if review not in selected and len(selected) < limit:
            selected.append(review)
    if len(selected) < 2:
        raise ValueError("At least two unique, informative reviews are required for question answers.")
    return overall, selected


def validate_question_results(
    values: object,
    questions: list[str],
    evidence_by_id: dict[str, Review],
) -> list[dict[str, Any]]:
    if not isinstance(values, list) or len(values) != len(questions):
        raise ValueError("The model returned the wrong number of question answers")
    indexed = {str(value.get("row_key")): value for value in values if isinstance(value, dict)}
    results: list[dict[str, Any]] = []
    for index, question in enumerate(questions):
        value = indexed.get(str(index))
        if not value or str(value.get("question", "")).strip() != question:
            raise ValueError(f"Invalid question answer at row {index}")
        answer = str(value.get("answer") or "").strip()
        review_ids = value.get("sample_review_ids")
        if not answer or not isinstance(review_ids, list) or len(review_ids) != 2:
            raise ValueError(f"Question row {index} needs an answer and two sample reviews")
        review_ids = [str(review_id) for review_id in review_ids]
        if len(set(review_ids)) != 2 or any(review_id not in evidence_by_id for review_id in review_ids):
            raise ValueError(f"Question row {index} contains invalid sample review IDs")
        results.append({
            "question": question,
            "answer": answer,
            "samples": [
                {"review_id": review_id, "source": evidence_by_id[review_id].source,
                 "sentiment": evidence_by_id[review_id].sentiment,
                 "text": evidence_by_id[review_id].text}
                for review_id in review_ids
            ],
        })
    return results


def generate_question_batch(
    client: Any,
    model: str,
    questions: list[str],
    dominant: str,
    evidence: list[Review],
    max_retries: int,
) -> list[dict[str, Any]]:
    evidence_by_id = {review.review_id: review for review in evidence}
    payload = {
        "dominant_sentiment": dominant,
        "questions": [{"row_key": str(index), "question": question}
                      for index, question in enumerate(questions)],
        "reviews": [
            {"review_id": review.review_id, "source": review.source,
             "category": review.primary_category, "text": review.text[:650]}
            for review in evidence
        ],
    }
    try:
        parsed = call_json(client, model, QUESTION_PROMPT, payload, max_retries)
        return validate_question_results(parsed.get("results"), questions, evidence_by_id)
    except Exception:
        if len(questions) == 1:
            raise
        middle = len(questions) // 2
        return (generate_question_batch(client, model, questions[:middle], dominant,
                                        evidence, max_retries)
                + generate_question_batch(client, model, questions[middle:], dominant,
                                          evidence, max_retries))


def load_cached_question_answers(report_path: Path) -> list[dict[str, Any]] | None:
    if not report_path.is_file():
        return None
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if report.get("question_analysis_version") != QUESTION_ANALYSIS_VERSION:
        return None
    cached = report.get("questions_answered")
    if not isinstance(cached, list) or [item.get("question") for item in cached] != list(QUESTIONS):
        return None
    return cached


def build_question_answers(
    reviews: list[Review], client: Any, model: str, batch_size: int,
    evidence_size: int, max_retries: int,
) -> tuple[str, list[dict[str, Any]]]:
    dominant, evidence = select_question_evidence(reviews, evidence_size)
    answers: list[dict[str, Any]] = []
    questions = list(QUESTIONS)
    for start in range(0, len(questions), batch_size):
        answers.extend(generate_question_batch(
            client, model, questions[start:start + batch_size], dominant,
            evidence, max_retries,
        ))
        print(f"Answered {min(start + batch_size, len(questions))}/{len(questions)} questions")
    for answer in answers:
        answer["dominant_sentiment"] = dominant
    return dominant, answers


def write_outputs(output: Path, reviews: list[Review], categories: list[dict[str, Any]],
                  opportunities: list[dict[str, Any]],
                  question_answers: list[dict[str, Any]]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "overview": {"total_reviews": len(reviews),
                     "overall_dominant_sentiment": dominant_sentiment(Counter(r.sentiment for r in reviews))},
        "categories": categories, "wishlist_opportunities": opportunities,
        "question_analysis_version": QUESTION_ANALYSIS_VERSION,
        "questions_answered": question_answers,
        "methodology_note": "Review shares are evidence signals, not measured conversion rates.",
    }
    (output / "review_analysis.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output / "review_category_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["category", "label", "review_count", "review_share_percent",
                  "positive_count", "positive_percent", "negative_count", "negative_percent",
                  "neutral_count", "neutral_percent", "dominant_sentiment", "summary"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for item in categories:
            writer.writerow({"category": item["category"], "label": item["label"],
                "review_count": item["review_count"], "review_share_percent": item["review_share_percent"],
                **{f"{s}_{k}": item["sentiments"][s][k] for s in SENTIMENTS for k in ("count", "percent")},
                "dominant_sentiment": item["dominant_sentiment"], "summary": item["summary"]})
    with (output / "classified_reviews.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["review_id", "source", "primary_category", "sentiment", "confidence",
                  "wishlist_relevant", "wishlist_barrier", "purchase_intent",
                  "purchase_outcome", "severity", "text"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for review in reviews:
            row = asdict(review); row.pop("chunks"); writer.writerow(row)
    with (output / "wishlist_opportunities.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["rank", "barrier", "mentions", "wishlist_review_share_percent",
                  "negative_rate_percent", "abandonment_rate_percent", "high_intent_rate_percent",
                  "high_severity_rate_percent", "opportunity_score", "suggested_improvement"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for item in opportunities:
            writer.writerow({key: item[key] for key in fields})


def print_results(categories: list[dict[str, Any]], opportunities: list[dict[str, Any]]) -> None:
    print("\nCATEGORY AND SENTIMENT SUMMARY")
    for item in categories:
        print(f"\n{item['label']}: {item['review_count']} reviews | "
              f"DOMINANT: {item['dominant_sentiment'].upper()}")
        print(item["summary"])
    print("\nWISHLIST IMPROVEMENT OPPORTUNITIES")
    for item in opportunities:
        print(f"{item['rank']}. {item['barrier']} — score {item['opportunity_score']}: "
              f"{item['suggested_improvement']}")


def main() -> None:
    args = parse_args()
    if chromadb is None:
        raise ModuleNotFoundError("Install ChromaDB: python -m pip install chromadb")
    if Groq is None:
        raise ModuleNotFoundError("Install Groq: python -m pip install groq")
    if not args.persist_directory.is_dir():
        raise FileNotFoundError(f"ChromaDB directory not found: {args.persist_directory}")
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Set GROQ_API_KEY in .env or the environment")
    for name in ("read_batch_size", "classification_batch_size", "max_retries",
                 "summary_sample_size", "samples_per_opportunity",
                 "question_batch_size", "question_evidence_size"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")

    client = Groq(api_key=api_key)
    database = chromadb.PersistentClient(path=str(args.persist_directory))
    try:
        collection = database.get_collection(args.collection)
    except Exception as error:
        raise ValueError(f"Chroma collection not found: {args.collection}") from error
    reviews = load_reviews(collection, args.read_batch_size)
    if not reviews:
        raise ValueError("No non-empty reviews were found in the collection")
    classify_reviews(collection, reviews, client, args.model,
                     args.classification_batch_size, args.max_retries,
                     args.overwrite_analysis)
    categories = build_category_results(reviews, client, args.model,
                                        args.summary_sample_size, args.max_retries)
    opportunities = rank_wishlist_opportunities(reviews, args.samples_per_opportunity)
    report_path = args.output_directory / "review_analysis.json"
    question_answers = None if args.regenerate_question_answers else load_cached_question_answers(report_path)
    if question_answers is None:
        _, question_answers = build_question_answers(
            reviews, client, args.model, args.question_batch_size,
            args.question_evidence_size, args.max_retries,
        )
    else:
        print("Reusing cached question answers from the existing report.")
    write_outputs(args.output_directory, reviews, categories, opportunities, question_answers)
    print_results(categories, opportunities)
    print(f"\nReports saved in: {args.output_directory.resolve()}")


if __name__ == "__main__":
    main()
