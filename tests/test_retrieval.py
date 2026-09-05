from review_vector_pipeline.retrieval import RetrievedReview, build_llm_input


def test_llm_input_contains_question_and_numbered_evidence():
    reviews = [
        RetrievedReview(
            rank=1,
            review_id="abc",
            text="Delivery was late",
            distance=0.2,
            metadata={"source": "play_store", "rating": 2.0},
        )
    ]
    prompt = build_llm_input("What delivery problems are reported?", reviews)
    assert "What delivery problems are reported?" in prompt
    assert "[Review 1] ID: abc" in prompt
    assert "Delivery was late" in prompt
    assert "Rating: 2.0" in prompt
