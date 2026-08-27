import chromadb
from collections import defaultdict


CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "cleaned_reviews"

client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_collection(COLLECTION_NAME)

results = collection.get(
    include=["metadatas"],
)

source_review_ids = defaultdict(set)

for record_id, metadata in zip(
    results["ids"],
    results["metadatas"],
):
    metadata = metadata or {}

    source = str(
        metadata.get("source", "unknown")
    ).strip().lower()

    review_id = str(
        metadata.get("review_id", record_id)
    )

    source_review_ids[source].add(review_id)

source_labels = {
    "play_store": "Play Store",
    "app_store": "App Store",
    "reddit": "Reddit",
    "unknown": "Unknown",
}

print()
print(f"{'SOURCE':<20}{'REVIEWS':>12}")
print("-" * 32)

total = 0

for source in [
    "play_store",
    "app_store",
    "reddit",
]:
    count = len(source_review_ids.get(source, set()))
    total += count

    print(
        f"{source_labels[source]:<20}"
        f"{count:>12,}"
    )

# Display additional source values, if present.
known_sources = {
    "play_store",
    "app_store",
    "reddit",
}

for source, review_ids in sorted(
    source_review_ids.items()
):
    if source not in known_sources:
        count = len(review_ids)
        total += count

        label = source_labels.get(
            source,
            source.replace("_", " ").title(),
        )

        print(f"{label:<20}{count:>12,}")

print("-" * 32)
print(f"{'Total':<20}{total:>12,}")