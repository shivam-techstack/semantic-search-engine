"""
evaluation.py — Precision, Recall, F1-score evaluation.

Contains a manually labelled test set of 10 queries with their relevant document IDs.
Computes metrics for both TF-IDF (keyword) and Word2Vec (semantic) search.
"""

from search import semantic_search, keyword_search

# ---------------------------------------------------------------------------
# Manually labelled test set
# Each entry: (query_string, set_of_relevant_doc_ids)
# These were created by reading the documents and judging relevance manually.
# ---------------------------------------------------------------------------
TEST_QUERIES = [
    (
        "How can doctors identify and diagnose diseases?",
        {2, 5, 30, 67}   # AI in healthcare, computer vision, malware (disease detection metaphor), IDS
    ),
    (
        "machine learning classification algorithms",
        {6, 7, 9, 10, 11, 57}
    ),
    (
        "protecting computer systems from hackers",
        {26, 27, 29, 30, 67, 68}
    ),
    (
        "storing and processing large amounts of data",
        {24, 31, 32, 53, 64}
    ),
    (
        "neural networks for image recognition",
        {13, 14, 15, 5}
    ),
    (
        "internet and network communication protocols",
        {42, 43, 44, 46, 47}
    ),
    (
        "text analysis and document retrieval",
        {36, 37, 38, 39, 40, 41}
    ),
    (
        "predicting future values from historical patterns",
        {60, 7, 8, 58}
    ),
    (
        "securing user data and privacy",
        {26, 28, 35, 63, 68, 74}
    ),
    (
        "understanding and generating human language",
        {4, 79, 16, 19}
    ),
]


def precision_at_k(retrieved_ids: list[int], relevant_ids: set[int], k: int) -> float:
    """
    Precision@K: out of the top-K retrieved documents, what fraction are relevant?
    P@K = |relevant ∩ top-K retrieved| / K
    """
    top_k = retrieved_ids[:k]
    hits  = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / k if k > 0 else 0.0


def recall_at_k(retrieved_ids: list[int], relevant_ids: set[int], k: int) -> float:
    """
    Recall@K: out of all relevant documents, what fraction did we retrieve in top-K?
    R@K = |relevant ∩ top-K retrieved| / |relevant|
    """
    top_k = retrieved_ids[:k]
    hits  = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / len(relevant_ids) if relevant_ids else 0.0


def f1_score(precision: float, recall: float) -> float:
    """F1 = harmonic mean of precision and recall."""
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def evaluate_method(search_fn, k: int = 5) -> dict:
    """
    Evaluate a search function (semantic or keyword) on all test queries.

    Returns averaged Precision@K, Recall@K, and F1-score.
    """
    precisions = []
    recalls    = []
    f1s        = []
    per_query  = []

    for query, relevant_ids in TEST_QUERIES:
        result      = search_fn(query, top_k=k)
        retrieved   = [r["id"] for r in result["results"]]

        p = precision_at_k(retrieved, relevant_ids, k)
        r = recall_at_k(retrieved, relevant_ids, k)
        f = f1_score(p, r)

        precisions.append(p)
        recalls.append(r)
        f1s.append(f)

        per_query.append({
            "query":       query,
            "precision":   round(p, 4),
            "recall":      round(r, 4),
            "f1":          round(f, 4),
            "retrieved":   retrieved,
            "relevant":    list(relevant_ids),
        })

    return {
        "avg_precision": round(sum(precisions) / len(precisions), 4),
        "avg_recall":    round(sum(recalls)    / len(recalls),    4),
        "avg_f1":        round(sum(f1s)        / len(f1s),        4),
        "k":             k,
        "per_query":     per_query,
    }


def run_full_evaluation(k: int = 5) -> dict:
    """
    Run evaluation on both search methods and return combined results.
    Called by the Flask /api/evaluate endpoint.
    """
    print(f"[evaluation] Running evaluation at K={k}...")

    semantic_results = evaluate_method(semantic_search, k=k)
    keyword_results  = evaluate_method(keyword_search,  k=k)

    print(f"[evaluation] Semantic — P:{semantic_results['avg_precision']:.4f}  "
          f"R:{semantic_results['avg_recall']:.4f}  F1:{semantic_results['avg_f1']:.4f}")
    print(f"[evaluation] Keyword  — P:{keyword_results['avg_precision']:.4f}  "
          f"R:{keyword_results['avg_recall']:.4f}  F1:{keyword_results['avg_f1']:.4f}")

    return {
        "k":       k,
        "semantic": semantic_results,
        "keyword":  keyword_results,
        "queries":  [q for q, _ in TEST_QUERIES],
    }


if __name__ == "__main__":
    results = run_full_evaluation(k=5)
    print("\n=== Evaluation Results (K=5) ===")
    print(f"Semantic Search  — Precision: {results['semantic']['avg_precision']:.4f}  "
          f"Recall: {results['semantic']['avg_recall']:.4f}  "
          f"F1: {results['semantic']['avg_f1']:.4f}")
    print(f"Keyword Search   — Precision: {results['keyword']['avg_precision']:.4f}  "
          f"Recall: {results['keyword']['avg_recall']:.4f}  "
          f"F1: {results['keyword']['avg_f1']:.4f}")
