"""
DeepEval Hallucination Check for the CRAG Research Assistant.

Measures:
  - HallucinationMetric : % of claims in the answer NOT present in any retrieved context chunk.
  - AnswerRelevancyMetric: How well the answer addresses the question.

Usage:
    python eval/run_deepeval.py

Requirements:
    pip install deepeval
"""

import json
import sys
import os

# Allow running from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepeval.metrics import HallucinationMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

from modules import vector_store
from modules.llm_generator import generate


# ── Load golden dataset ──────────────────────────────────────────────────────

GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "golden_dataset.json")

with open(GOLDEN_PATH, encoding="utf-8") as f:
    golden_dataset = [q for q in json.load(f) if not q.get("_comment")]

print(f"Loaded {len(golden_dataset)} golden questions.\n")


# ── Build evaluation rows ────────────────────────────────────────────────────

def build_eval_row(q: dict) -> dict:
    store = vector_store.get_vector_store()
    results = store.similarity_search_with_score(q["question"], k=5)
    docs = [
        {"text": doc.page_content, "score": round(1.0 / (1.0 + float(score)), 4)}
        for doc, score in results
    ]
    context_texts = [d["text"] for d in docs]
    combined_context = "\n\n---\n\n".join(context_texts)
    result = generate(q["question"], combined_context)
    return {
        "question": q["question"],
        "answer": result["answer"],
        "contexts": context_texts,
        "ground_truth": q["ground_truth"],
    }


print("Running pipeline on all questions...")
rows = [build_eval_row(q) for q in golden_dataset]


# ── Run DeepEval metrics ─────────────────────────────────────────────────────

hallucination_metric = HallucinationMetric(threshold=0.5)
relevancy_metric = AnswerRelevancyMetric(threshold=0.7)

results_log = []

print("\n=== DeepEval Hallucination + Relevancy Report ===\n")
for row in rows:
    test_case = LLMTestCase(
        input=row["question"],
        actual_output=row["answer"],
        context=row["contexts"],
        expected_output=row["ground_truth"],
    )

    hallucination_metric.measure(test_case)
    relevancy_metric.measure(test_case)

    entry = {
        "question": row["question"][:80] + "...",
        "hallucination_score": round(hallucination_metric.score, 3),
        "hallucination_reason": hallucination_metric.reason,
        "relevancy_score": round(relevancy_metric.score, 3),
        "relevancy_reason": relevancy_metric.reason,
        "hallucination_passed": hallucination_metric.is_successful(),
        "relevancy_passed": relevancy_metric.is_successful(),
    }
    results_log.append(entry)

    status = "✅" if entry["hallucination_passed"] else "❌"
    print(f"{status} Hallucination: {entry['hallucination_score']:.2f}  |  "
          f"Relevancy: {entry['relevancy_score']:.2f}")
    print(f"   Q: {entry['question']}")
    if not entry["hallucination_passed"]:
        print(f"   ↳ Reason: {entry['hallucination_reason']}\n")

# Summary
total = len(results_log)
passed_halluc = sum(1 for r in results_log if r["hallucination_passed"])
passed_relev = sum(1 for r in results_log if r["relevancy_passed"])
print(f"\nSummary: Hallucination passed {passed_halluc}/{total} | Relevancy passed {passed_relev}/{total}")

# Save results
output_path = os.path.join(os.path.dirname(__file__), "deepeval_results.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results_log, f, indent=2, default=str)
print(f"Detailed results saved to: {output_path}")
