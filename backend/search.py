"""
search.py — Core search logic.

Provides:
  - semantic_search()          Word2Vec averaged vectors + cosine similarity
  - keyword_search()           TF-IDF + cosine similarity
  - expanded_semantic_search() Word2Vec query expansion (from google-but-smarter)
  - compare_search()           Side-by-side: semantic vs keyword
  - _inverted_index            Inverted index for term lookup
  - _query_to_vector()         Exposed so rocchio.py can call it
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from preprocessing import preprocess
from database import get_all_documents, get_document_by_id

# ---------------------------------------------------------------------------
# Module-level state (populated by app.py at startup via set_engine_state)
# ---------------------------------------------------------------------------
_w2v_model         = None
_doc_vectors       = None   # numpy array shape (N, 100)
_doc_ids           = None   # numpy array shape (N,)
_tfidf_vectorizer  = None
_tfidf_matrix      = None
_tfidf_doc_ids     = None
_inverted_index    = {}     # word → [doc_id, doc_id, ...]


def set_engine_state(model, doc_vectors, doc_ids):
    """
    Called once at startup. Injects model + vectors and builds all indexes.
    """
    global _w2v_model, _doc_vectors, _doc_ids
    _w2v_model   = model
    _doc_vectors = doc_vectors
    _doc_ids     = doc_ids

    _build_tfidf_index()
    _build_inverted_index()


# ---------------------------------------------------------------------------
# Index builders
# ---------------------------------------------------------------------------

def _build_tfidf_index():
    """Build a TF-IDF matrix over all documents for keyword search."""
    global _tfidf_vectorizer, _tfidf_matrix, _tfidf_doc_ids

    documents = get_all_documents()
    corpus, ids = [], []

    for doc in documents:
        tokens = preprocess(doc["title"] + " " + doc["content"])
        corpus.append(" ".join(tokens))
        ids.append(doc["id"])

    _tfidf_vectorizer = TfidfVectorizer()
    _tfidf_matrix     = _tfidf_vectorizer.fit_transform(corpus)
    _tfidf_doc_ids    = ids
    print(f"[search] TF-IDF index built. Matrix shape: {_tfidf_matrix.shape}")


def _build_inverted_index():
    """
    Build an inverted index: word → sorted list of document IDs.
    Inspired by the custom inverted index in google-but-smarter.
    Used for the /api/index/lookup endpoint and fast term-presence checks.
    """
    global _inverted_index
    _inverted_index = {}

    documents = get_all_documents()
    for doc in documents:
        tokens = set(preprocess(doc["title"] + " " + doc["content"]))
        for token in tokens:
            if token not in _inverted_index:
                _inverted_index[token] = []
            _inverted_index[token].append(doc["id"])

    # Sort each posting list for consistency
    for token in _inverted_index:
        _inverted_index[token].sort()

    print(f"[search] Inverted index built. {len(_inverted_index)} unique terms.")


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _query_to_vector(query: str):
    """
    Convert a raw query string to an averaged Word2Vec vector.
    Exposed at module level so rocchio.py can import it directly.
    Returns numpy array or None if no tokens found in vocabulary.
    """
    from embeddings import document_to_vector
    tokens = preprocess(query)
    if not tokens:
        return None
    return document_to_vector(tokens, _w2v_model)


def _format_result(doc_id: int, score: float) -> dict | None:
    """Fetch document from DB and format it as a result dict."""
    doc = get_document_by_id(doc_id)
    if doc is None:
        return None
    return {
        "id":        doc_id,
        "title":     doc["title"],
        "category":  doc["category"],
        "preview":   doc["preview"],
        "score":     round(score, 4),
        "score_pct": f"{score * 100:.1f}%",
    }


# ---------------------------------------------------------------------------
# 1. Semantic Search (Word2Vec averaged vectors)
# ---------------------------------------------------------------------------

def semantic_search(query: str, top_k: int = 10, min_score: float = 0.0) -> dict:
    """
    Semantic search using Word2Vec embeddings + cosine similarity.
    Each document is represented as the average of its word vectors.
    The query is converted the same way and compared to all doc vectors.
    """
    if _w2v_model is None:
        raise RuntimeError("Search engine not initialized. Call set_engine_state() first.")

    query_vector = _query_to_vector(query)
    if query_vector is None:
        return {"query": query, "results": [], "result_count": 0,
                "search_type": "semantic",
                "error": "Query terms not found in vocabulary."}

    similarities  = cosine_similarity(query_vector.reshape(1, -1), _doc_vectors)[0]
    ranked_indices = np.argsort(similarities)[::-1]

    results = []
    for idx in ranked_indices:
        score = float(similarities[idx])
        if score < min_score:
            break
        r = _format_result(int(_doc_ids[idx]), score)
        if r:
            results.append(r)
        if len(results) >= top_k:
            break

    return {"query": query, "results": results,
            "result_count": len(results), "search_type": "semantic"}


# ---------------------------------------------------------------------------
# 2. Keyword Search (TF-IDF)
# ---------------------------------------------------------------------------

def keyword_search(query: str, top_k: int = 10, min_score: float = 0.0) -> dict:
    """
    Traditional keyword search using TF-IDF weighted vectors + cosine similarity.
    Only finds documents containing the exact (preprocessed) query terms.
    """
    if _tfidf_vectorizer is None:
        raise RuntimeError("TF-IDF index not built. Call set_engine_state() first.")

    tokens       = preprocess(query)
    query_string = " ".join(tokens)

    if not query_string.strip():
        return {"query": query, "results": [], "result_count": 0,
                "search_type": "keyword",
                "error": "No valid query terms after preprocessing."}

    query_vec    = _tfidf_vectorizer.transform([query_string])
    similarities = cosine_similarity(query_vec, _tfidf_matrix)[0]
    ranked_indices = np.argsort(similarities)[::-1]

    results = []
    for idx in ranked_indices:
        score = float(similarities[idx])
        if score < min_score:
            break
        r = _format_result(_tfidf_doc_ids[idx], score)
        if r:
            results.append(r)
        if len(results) >= top_k:
            break

    return {"query": query, "results": results,
            "result_count": len(results), "search_type": "keyword"}


# ---------------------------------------------------------------------------
# 3. Word2Vec Query Expansion Search (inspired by google-but-smarter)
# ---------------------------------------------------------------------------

def expanded_semantic_search(query: str, top_k: int = 10,
                              expand_topn: int = 3,
                              similarity_threshold: float = 0.65) -> dict:
    """
    Word2Vec Query Expansion Search.

    Instead of using only the query words, we find semantically similar words
    from the Word2Vec vocabulary and add them to the query before searching.

    Example:
      Query:    "disease detection"
      Expanded: "disease detection diagnosis medical identify symptom"

    This bridges the vocabulary gap — if documents use different words than
    the query but mean the same thing, they can still be retrieved.

    Inspired by the Word2Vec expansion mode in google-but-smarter.
    """
    from embeddings import document_to_vector

    tokens = preprocess(query)
    if not tokens:
        return {"query": query, "results": [], "result_count": 0,
                "search_type": "expanded", "expanded_terms": []}

    # Expand: for each token, find top similar words from Word2Vec vocabulary
    expanded_tokens = list(tokens)
    added_terms = []

    for token in tokens:
        if token in _w2v_model.wv:
            similar_words = _w2v_model.wv.most_similar(token, topn=expand_topn)
            for word, score in similar_words:
                if score >= similarity_threshold and word not in expanded_tokens:
                    expanded_tokens.append(word)
                    added_terms.append(f"{word} ({score:.2f})")

    # Generate vector from expanded token list
    query_vector = document_to_vector(expanded_tokens, _w2v_model)
    if query_vector is None:
        return {"query": query, "results": [], "result_count": 0,
                "search_type": "expanded", "expanded_terms": added_terms}

    similarities   = cosine_similarity(query_vector.reshape(1, -1), _doc_vectors)[0]
    ranked_indices = np.argsort(similarities)[::-1]

    results = []
    for idx in ranked_indices[:top_k]:
        r = _format_result(int(_doc_ids[idx]), float(similarities[idx]))
        if r:
            results.append(r)

    return {
        "query":          query,
        "expanded_terms": added_terms,
        "original_tokens": tokens,
        "expanded_tokens": expanded_tokens,
        "results":         results,
        "result_count":    len(results),
        "search_type":     "expanded",
    }


# ---------------------------------------------------------------------------
# 4. Comparison: Semantic vs Keyword (existing, unchanged)
# ---------------------------------------------------------------------------

def compare_search(query: str, top_k: int = 5) -> dict:
    """Run semantic and keyword search side-by-side."""
    semantic = semantic_search(query, top_k=top_k)
    keyword  = keyword_search(query,  top_k=top_k)
    return {
        "query":    query,
        "semantic": semantic["results"],
        "keyword":  keyword["results"],
    }


# ---------------------------------------------------------------------------
# 5. All-modes comparison (new: semantic vs keyword vs expanded vs rocchio)
# ---------------------------------------------------------------------------

def compare_all_modes(query: str, top_k: int = 5) -> dict:
    """
    Run all four search modes and return results side-by-side.
    Used by the enhanced comparison page in the frontend.
    """
    from rocchio import rocchio_search

    semantic = semantic_search(query,          top_k=top_k)
    keyword  = keyword_search(query,           top_k=top_k)
    expanded = expanded_semantic_search(query, top_k=top_k)
    rocchio  = rocchio_search(query,           top_k=top_k)

    return {
        "query":    query,
        "semantic": semantic["results"],
        "keyword":  keyword["results"],
        "expanded": expanded["results"],
        "rocchio":  rocchio["results"],
        "expanded_terms": expanded.get("expanded_terms", []),
    }
