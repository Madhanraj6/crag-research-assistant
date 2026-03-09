"""
Confidence Scoring Module — computes a composite confidence score for the response.

Formula:
    confidence = 0.5 * similarity_score
              + 0.2 * retrieval_consistency
              + 0.2 * llm_self_score
              - 0.1 * correction_penalty
"""


def compute(
    avg_similarity: float,
    retrieval_scores: list[float],
    llm_self_score: float,
    correction_attempts: int,
    used_web_fallback: bool,
) -> dict:
    """
    Compute a composite confidence score.

    Args:
        avg_similarity: Average cosine similarity from vector retrieval (0-1).
        retrieval_scores: List of individual document scores.
        llm_self_score: LLM's self-evaluation score (0-1).
        correction_attempts: Number of query reformulation attempts made.
        used_web_fallback: Whether web search was triggered.

    Returns:
        {
            "confidence": float (0-1),
            "breakdown": dict with component scores,
            "level": str ("high" / "medium" / "low")
        }
    """
    # --- Similarity component (0-1) ---
    similarity_component = min(1.0, max(0.0, avg_similarity))

    # --- Retrieval consistency (0-1) ---
    # Measures how consistent the retrieval scores are (less variance = more consistent)
    if len(retrieval_scores) > 1:
        mean = sum(retrieval_scores) / len(retrieval_scores)
        variance = sum((s - mean) ** 2 for s in retrieval_scores) / len(retrieval_scores)
        # Normalize: low variance → high consistency
        consistency = max(0.0, 1.0 - (variance * 4))  # scale factor
    elif len(retrieval_scores) == 1:
        consistency = retrieval_scores[0]
    else:
        consistency = 0.0

    # --- LLM self-score (0-1) ---
    llm_component = min(1.0, max(0.0, llm_self_score))

    # --- Correction penalty ---
    # Each correction attempt reduces confidence by 0.1 (max penalty)
    penalty = min(1.0, correction_attempts * 0.5)

    # Additional penalty if web fallback was needed
    if used_web_fallback:
        penalty = min(1.0, penalty + 0.3)

    # --- Composite score ---
    confidence = (
        0.5 * similarity_component
        + 0.2 * consistency
        + 0.2 * llm_component
        - 0.1 * penalty
    )

    # Clamp to [0, 1]
    confidence = max(0.0, min(1.0, round(confidence, 4)))

    # Determine level
    if confidence >= 0.75:
        level = "high"
    elif confidence >= 0.5:
        level = "medium"
    else:
        level = "low"

    return {
        "confidence": confidence,
        "level": level,
        "breakdown": {
            "similarity": round(similarity_component, 4),
            "consistency": round(consistency, 4),
            "llm_self_score": round(llm_component, 4),
            "correction_penalty": round(penalty, 4),
            "web_fallback_used": used_web_fallback,
            "correction_attempts": correction_attempts,
        },
    }
