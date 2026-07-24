from unittest.mock import Mock

from evaluation.evaluator import run_evaluation


def test_retrieval_only_evaluation_skips_generation():
    pipeline = Mock()
    pipeline.ask.return_value = {
        "answer": "",
        "sources": [{"source": "document.pdf", "page": 1}],
        "retrieval_ms": 12.0,
        "generation_ms": 0.0,
        "total_ms": 12.0,
    }
    cases = [
        {
            "id": "q01",
            "question": "Soru",
            "reference_answer": "Cevap",
            "relevant_sources": [{"source": "document.pdf", "page": 1}],
        }
    ]

    result = run_evaluation(
        pipeline=pipeline,
        test_cases=cases,
        top_k=1,
        generate_answers=False,
    )

    assert result.iloc[0]["hit_rate_at_k"] == 1.0
    assert result.iloc[0]["answer"] == ""
    assert pipeline.ask.call_args.kwargs["generate_answer"] is False
