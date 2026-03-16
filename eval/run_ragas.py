"""
RAGAS Evaluation Harness for the CRAG Research Assistant.

Measures:
  - Faithfulness         : Are claims grounded in the retrieved context?
  - Answer Relevancy     : Does the answer address the question?
  - Context Precision    : Are retrieved docs actually useful?
  - Context Recall       : Did retrieval miss relevant docs?

Usage:
    python eval/run_ragas.py

Requirements:
    pip install ragas datasets
"""

import json
import sys
import os

# Allow running from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from modules import vector_store
from modules.llm_generator import generate


# ── Load golden dataset ──────────────────────────────────────────────────────

GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "golden_dataset.json")

with open(GOLDEN_PATH, encoding="utf-8") as f:
    golden_dataset = [q for q in json.load(f) if not q.get("_comment")]

print(f"Loaded {len(golden_dataset)} golden questions.")


# ── Build evaluation rows ────────────────────────────────────────────────────

def build_eval_row(q: dict) -> dict:
    """Run one question through the full CRAG pipeline and collect RAGAS fields."""
    store = vector_store.get_vector_store()

    # Retrieve top-5 candidates (pre-rerank)
    results = store.similarity_search_with_score(q["question"], k=5)
    docs = [
        {"text": doc.page_content, "score": round(1.0 / (1.0 + float(score)), 4), "metadata": doc.metadata}
        for doc, score in results
    ]
    context_texts = [d["text"] for d in docs]
    combined_context = "\n\n---\n\n".join(context_texts)

    # Generate the answer (non-streaming path)
    result = generate(q["question"], combined_context)

    return {
        "question": q["question"],
        "answer": result["answer"],
        "contexts": context_texts,
        "ground_truth": q["ground_truth"],
    }


print("Running pipeline on all questions (this may take a few minutes)...")
rows = [build_eval_row(q) for q in golden_dataset]

dataset = Dataset.from_list(rows)


# ── Run RAGAS ────────────────────────────────────────────────────────────────

print("\nEvaluating with RAGAS metrics...")
result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
)

print("\n=== RAGAS Results ===")
print(result)

# Save to JSON for later comparison
output_path = os.path.join(os.path.dirname(__file__), "ragas_results.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(result.to_pandas().to_dict(orient="records"), f, indent=2, default=str)

print(f"\nDetailed results saved to: {output_path}")
