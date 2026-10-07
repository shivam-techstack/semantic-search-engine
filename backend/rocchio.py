"""
rocchio.py — Rocchio Pseudo-Relevance Feedback Algorithm.
Inspired by the Rocchio mode in google-but-smarter.

How it works:
  1. Run an initial semantic search to get top results.
  2. Treat those top results as "relevant" documents (pseudo-relevance assumption).
  3. Compute their centroid vector (average of their document vectors).
  4. Move the query vector TOWARD that centroid using weighted combination:

       new_query = alpha * original_query + beta * relevant_centroid

  5. Re-run cosine similarity with the improved query vector.

This automatically improves the query without asking the user for feedback.
Named after Gerard Salton (1971) — a foundational IR algorithm.

Parameters:
  ALPHA = 1.0   (how much to keep the original query)
  BETA  = 0.75  (how much to move toward relevant documents)
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

ALPHA = 1.0    # weight for original query vector
BETA  = 0.75   # weight for relevant document centroid


def rocchio_expand(query_vector: np.ndarray,
                   top_doc_vectors: np.ndarray,
                   alpha: float = ALPHA,
                   beta: float = BETA) -> np.ndarray:
    """
    Apply the Rocchio algorithm to produce an improved query vector.

    Parameters
    ----------
    query_vector    : Original query vector, shape (100,)
    top_doc_vectors : Vectors of top-K retrieved documents, shape (K, 100)
    alpha           : Weight for original query direction
    beta            : Weight for relevant document centroid direction

    Returns
    -------
    Improved query vector, shape (100,), L2-normalized.
    """
    if len(top_doc_vectors) == 0:
        return query_vector

    # Centroid = average of top document vectors
    relevant_centroid = np.mean(top_doc_vectors, axis=0)

    # Weighted combination: move query toward relevant documents
    new_query = alpha * query_vector + beta * relevant_centroid

    # L2 normalize so cosine similarity stays valid
    norm = np.linalg.norm(new_query)
    if norm > 0:
        new_query = new_query / norm

    return new_query


def rocchio_search(query: str, top_k: int = 10, feedback_k: int = 3) -> dict:
    """
    Full Rocchio pseudo-relevance feedback search.

    Steps:
      1. Run initial semantic search (get top feedback_k docs).
      2. Retrieve those documents' precomputed vectors.
      3. Apply Rocchio to expand the query vector.
      4. Re-rank ALL documents with the improved query vector.
      5. Return top_k results.

    Parameters
    ----------
    query      : User's search query
    top_k      : How many final results to return
    feedback_k : How many top results to use as pseudo-relevant feedback
    """
    # Import here to avoid circular imports at module load time
    from search import _w2v_model, _doc_vectors, _doc_ids, _query_to_vector
    from database import get_document_by_id

    if _w2v_model is None:
        raise RuntimeError("Search engine not initialized.")

    # Step 1: get original query vector
    query_vector = _query_to_vector(query)
    if query_vector is None:
        return {"query": query, "results": [], "result_count": 0,
                "search_type": "rocchio",
                "error": "Query terms not found in vocabulary."}

    # Step 2: initial cosine similarity pass to find top feedback_k docs
    initial_sims    = cosine_similarity(query_vector.reshape(1, -1), _doc_vectors)[0]
    top_indices     = np.argsort(initial_sims)[::-1][:feedback_k]
    top_doc_vectors = _doc_vectors[top_indices]   # shape (feedback_k, 100)

    # Step 3: Rocchio expansion — improve the query vector
    improved_vector = rocchio_expand(query_vector, top_doc_vectors)

    # Step 4: re-rank ALL documents with the improved query vector
    final_sims     = cosine_similarity(improved_vector.reshape(1, -1), _doc_vectors)[0]
    ranked_indices = np.argsort(final_sims)[::-1]

    # Step 5: collect top_k results
    results = []
    for idx in ranked_indices[:top_k]:
        score  = float(final_sims[idx])
        doc_id = int(_doc_ids[idx])
        doc    = get_document_by_id(doc_id)
        if doc:
            results.append({
                "id":        doc_id,
                "title":     doc["title"],
                "category":  doc["category"],
                "preview":   doc["preview"],
                "score":     round(score, 4),
                "score_pct": f"{score * 100:.1f}%",
            })

    return {
        "query":        query,
        "feedback_k":   feedback_k,
        "results":      results,
        "result_count": len(results),
        "search_type":  "rocchio",
    }
