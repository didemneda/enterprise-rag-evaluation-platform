import json
import pandas as pd

from evaluation.retrieval_metrics import (
    hit_rate_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def load_test_cases(path):
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def run_evaluation(
    pipeline,
    test_cases,
    top_k=5,
    minimum_score=0.0,
    generate_answers=False,
    use_llm_judge=False,
):
    rows = []

    for case in test_cases:
        result = pipeline.ask(
            case["question"],
            top_k=top_k,
            minimum_score=minimum_score,
            generate_answer=generate_answers or use_llm_judge,
        )

        retrieved = result["sources"]
        relevant = case.get("relevant_sources", [])

        row = {
            "id": case.get("id", ""),
            "question": case["question"],
            "answer": result["answer"],
            "reference_answer": case.get("reference_answer", ""),
            "precision_at_k": precision_at_k(retrieved, relevant, top_k),
            "recall_at_k": recall_at_k(retrieved, relevant, top_k),
            "reciprocal_rank": reciprocal_rank(retrieved, relevant),
            "hit_rate_at_k": hit_rate_at_k(retrieved, relevant, top_k),
            "retrieval_ms": result["retrieval_ms"],
            "generation_ms": result["generation_ms"],
            "total_ms": result["total_ms"],
        }

        if use_llm_judge:
            context = "\n\n".join(
                source["content"] for source in retrieved
            )
            row.update(
                pipeline.llm.judge(
                    question=case["question"],
                    answer=result["answer"],
                    context=context,
                    reference_answer=case.get("reference_answer", ""),
                )
            )

        rows.append(row)

    return pd.DataFrame(rows)
