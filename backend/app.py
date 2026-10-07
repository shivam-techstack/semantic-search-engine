"""
app.py — Flask backend for the Semantic Search Engine.

Routes:
  GET  /                              Serve the frontend
  GET  /api/search?q=&type=&k=        Main search (semantic / keyword / expanded / rocchio)
  GET  /api/compare?q=&k=             Semantic vs keyword side-by-side
  GET  /api/compare/all?q=&k=         All 4 modes side-by-side (new)
  GET  /api/evaluate?k=               Precision/Recall/F1 evaluation
  GET  /api/similar/<id>?k=           More-Like-This (new, from google-but-smarter)
  GET  /api/index/lookup?word=        Inverted index term lookup (new)
  GET  /api/documents                 List all documents
  GET  /api/document/<id>             Get single document
  GET  /api/status                    Health check / engine info
"""

import os
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
from document_loader import load_documents
from embeddings      import initialize_embeddings
from search          import (set_engine_state, semantic_search, keyword_search,
                              compare_search, expanded_semantic_search,
                              compare_all_modes, _inverted_index)
from rocchio         import rocchio_search
from evaluation      import run_full_evaluation
from database        import get_all_documents, get_document_by_id, document_count

print("[app] Starting Semantic Search Engine...")

load_documents()
w2v_model, doc_vectors, doc_ids = initialize_embeddings()
set_engine_state(w2v_model, doc_vectors, doc_ids)

print("[app] All systems ready.\n")

# ---------------------------------------------------------------------------
# Flask setup
# ---------------------------------------------------------------------------
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
CORS(app)


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


# ---------------------------------------------------------------------------
# Main Search — now supports 4 modes
# ---------------------------------------------------------------------------

@app.route("/api/search", methods=["GET"])
def search():
    """
    Unified search endpoint.

    Query parameters:
      q     : search query (required)
      type  : 'semantic' | 'keyword' | 'expanded' | 'rocchio'  (default: semantic)
      k     : number of results, 1–20  (default: 10)
    """
    query       = request.args.get("q", "").strip()
    search_type = request.args.get("type", "semantic").lower()
    k           = min(int(request.args.get("k", 10)), 20)

    if not query:
        return jsonify({"error": "Query parameter 'q' is required."}), 400
    if len(query) > 500:
        return jsonify({"error": "Query too long (max 500 characters)."}), 400
    if search_type not in ("semantic", "keyword", "expanded", "rocchio"):
        return jsonify({"error": "type must be: semantic | keyword | expanded | rocchio"}), 400

    try:
        if search_type == "semantic":
            result = semantic_search(query, top_k=k)
        elif search_type == "keyword":
            result = keyword_search(query, top_k=k)
        elif search_type == "expanded":
            result = expanded_semantic_search(query, top_k=k)
        else:
            result = rocchio_search(query, top_k=k)

        if result.get("result_count", 0) == 0:
            result["message"] = "No results found. Try different search terms."
        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": f"Search failed: {str(e)}"}), 500


# ---------------------------------------------------------------------------
# Comparison endpoints
# ---------------------------------------------------------------------------

@app.route("/api/compare", methods=["GET"])
def compare():
    """Semantic vs keyword — original comparison."""
    query = request.args.get("q", "").strip()
    k     = min(int(request.args.get("k", 5)), 10)
    if not query:
        return jsonify({"error": "Query parameter 'q' is required."}), 400
    try:
        return jsonify(compare_search(query, top_k=k)), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/compare/all", methods=["GET"])
def compare_all():
    """
    All 4 modes side-by-side: semantic / keyword / expanded / rocchio.
    New endpoint inspired by the 4-mode design of google-but-smarter.
    """
    query = request.args.get("q", "").strip()
    k     = min(int(request.args.get("k", 5)), 10)
    if not query:
        return jsonify({"error": "Query parameter 'q' is required."}), 400
    try:
        return jsonify(compare_all_modes(query, top_k=k)), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# More-Like-This  (inspired by /api/similar/<id> in google-but-smarter)
# ---------------------------------------------------------------------------

@app.route("/api/similar/<int:doc_id>", methods=["GET"])
def similar_documents(doc_id):
    """
    Find documents most similar to a given document.
    Uses the document's own precomputed vector as the query.
    Inspired by the similarity search feature in google-but-smarter.

    Query parameters:
      k : number of similar documents to return (default 5)
    """
    k = min(int(request.args.get("k", 5)), 10)

    # Find this document's index in the vectors array
    doc_indices = np.where(doc_ids == doc_id)[0]
    if len(doc_indices) == 0:
        return jsonify({"error": f"Document {doc_id} not found."}), 404

    # Use this document's vector as the query
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim
    this_vector  = doc_vectors[doc_indices[0]].reshape(1, -1)
    similarities = cos_sim(this_vector, doc_vectors)[0]
    ranked       = np.argsort(similarities)[::-1]

    results = []
    for idx in ranked:
        d_id = int(doc_ids[idx])
        if d_id == doc_id:
            continue  # skip the document itself
        score = float(similarities[idx])
        doc   = get_document_by_id(d_id)
        if doc:
            results.append({
                "id":        d_id,
                "title":     doc["title"],
                "category":  doc["category"],
                "preview":   doc["preview"],
                "score":     round(score, 4),
                "score_pct": f"{score * 100:.1f}%",
            })
        if len(results) >= k:
            break

    source_doc = get_document_by_id(doc_id)
    return jsonify({
        "source_id":       doc_id,
        "source_title":    source_doc["title"] if source_doc else "",
        "source_category": source_doc["category"] if source_doc else "",
        "similar":         results,
        "count":           len(results),
    }), 200


# ---------------------------------------------------------------------------
# Inverted Index Lookup  (new — shows classic IR concept in action)
# ---------------------------------------------------------------------------

@app.route("/api/index/lookup", methods=["GET"])
def index_lookup():
    """
    Look up which documents contain a specific word using the inverted index.
    Returns the posting list (list of document IDs) for a given term.

    Query parameters:
      word : the term to look up (required)
    """
    from search import _inverted_index
    from preprocessing import preprocess

    word = request.args.get("word", "").strip()
    if not word:
        return jsonify({"error": "Parameter 'word' is required."}), 400

    tokens = preprocess(word)
    if not tokens:
        return jsonify({"word": word, "token": None,
                        "doc_ids": [], "count": 0,
                        "message": "Word removed during preprocessing (stop word or too short)."}), 200

    token   = tokens[0]
    doc_ids_list = _inverted_index.get(token, [])

    # Fetch titles for display
    documents = []
    for did in doc_ids_list[:20]:  # cap at 20 for response size
        doc = get_document_by_id(did)
        if doc:
            documents.append({"id": did, "title": doc["title"],
                               "category": doc["category"]})

    return jsonify({
        "word":       word,
        "token":      token,
        "doc_ids":    doc_ids_list,
        "count":      len(doc_ids_list),
        "documents":  documents,
    }), 200


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@app.route("/api/evaluate", methods=["GET"])
def evaluate():
    """Precision/Recall/F1 on the manually labelled test set."""
    k = min(int(request.args.get("k", 5)), 10)
    try:
        return jsonify(run_full_evaluation(k=k)), 200
    except Exception as e:
        return jsonify({"error": f"Evaluation failed: {str(e)}"}), 500


# ---------------------------------------------------------------------------
# Document APIs
# ---------------------------------------------------------------------------

@app.route("/api/documents", methods=["GET"])
def documents():
    """Return all documents (lightweight — no full content)."""
    docs = get_all_documents()
    lightweight = [
        {"id": d["id"], "title": d["title"], "category": d["category"],
         "preview": d["preview"], "file_name": d["file_name"]}
        for d in docs
    ]
    return jsonify({"documents": lightweight, "count": len(lightweight)}), 200


@app.route("/api/document/<int:doc_id>", methods=["GET"])
def document(doc_id):
    """Return full details for one document."""
    doc = get_document_by_id(doc_id)
    if doc is None:
        return jsonify({"error": f"Document {doc_id} not found."}), 404
    return jsonify(doc), 200


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

@app.route("/api/status", methods=["GET"])
def status():
    """Health check."""
    from search import _inverted_index
    return jsonify({
        "status":        "ok",
        "documents":     document_count(),
        "vector_shape":  list(doc_vectors.shape),
        "vocab_size":    len(w2v_model.wv),
        "vector_size":   int(doc_vectors.shape[1]) if len(doc_vectors.shape) > 1 else 0,
        "index_terms":   len(_inverted_index),
        "search_modes":  ["semantic", "keyword", "expanded", "rocchio"],
        "message":       "Semantic Search Engine is running.",
    }), 200


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found."}), 404

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed."}), 405


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"[app] Server running at http://127.0.0.1:{port}")
    print(f"[app] Open http://127.0.0.1:{port} in your browser.\n")
    app.run(debug=True, port=port)
