"""
embeddings.py — Word2Vec model training and vector generation.

Design decisions:
  - We train Word2Vec on our own 80-document corpus.
    A pre-trained model like GoogleNews-300 is 1.6 GB — too large for a college project.
    Our smaller trained model captures domain-specific vocabulary (AI, ML, networking, etc.)
  - Vector size: 100 dimensions (good balance of quality and speed for a small corpus)
  - Document vector = average of all valid word vectors in the document
  - Precomputed document vectors are saved to vectors.npy so search is fast at runtime
"""

import os
import numpy as np
from gensim.models import Word2Vec

from preprocessing import preprocess
from database import get_all_documents

MODEL_DIR  = os.path.join(os.path.dirname(__file__), "data", "w2v_model")
MODEL_PATH = os.path.join(MODEL_DIR, "word2vec.model")
VECTORS_PATH = os.path.join(os.path.dirname(__file__), "data", "vectors.npy")
DOC_IDS_PATH = os.path.join(os.path.dirname(__file__), "data", "doc_ids.npy")

VECTOR_SIZE = 100   # dimensions per word vector
WINDOW      = 5     # context window size
MIN_COUNT   = 1     # include words that appear at least once (small corpus)
EPOCHS      = 50    # more epochs to compensate for small corpus size
WORKERS     = 4     # parallel training threads


def build_corpus(documents: list[dict]) -> list[list[str]]:
    """
    Preprocess all document texts into a list of token lists.
    This is the training corpus for Word2Vec.
    """
    corpus = []
    for doc in documents:
        # Combine title and content so the model learns title vocabulary too
        combined = doc["title"] + " " + doc["content"]
        tokens = preprocess(combined)
        if tokens:
            corpus.append(tokens)
    return corpus


def train_word2vec(documents: list[dict]) -> Word2Vec:
    """
    Train a Word2Vec model on the document corpus and save it to disk.
    Returns the trained model.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("[embeddings] Building training corpus...")
    corpus = build_corpus(documents)
    print(f"[embeddings] Corpus size: {len(corpus)} documents")

    print(f"[embeddings] Training Word2Vec (vector_size={VECTOR_SIZE}, epochs={EPOCHS})...")
    model = Word2Vec(
        sentences=corpus,
        vector_size=VECTOR_SIZE,
        window=WINDOW,
        min_count=MIN_COUNT,
        workers=WORKERS,
        epochs=EPOCHS,
        sg=1,           # Skip-gram (better for smaller corpora than CBOW)
    )

    model.save(MODEL_PATH)
    print(f"[embeddings] Model saved to {MODEL_PATH}")
    print(f"[embeddings] Vocabulary size: {len(model.wv)} words")
    return model


def load_word2vec() -> Word2Vec:
    """Load the saved Word2Vec model from disk."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Word2Vec model not found at {MODEL_PATH}. "
            "Run: python embeddings.py  to train it first."
        )
    return Word2Vec.load(MODEL_PATH)


def document_to_vector(tokens: list[str], model: Word2Vec) -> np.ndarray | None:
    """
    Convert a list of tokens into a single document vector by averaging
    the Word2Vec vectors of all tokens that exist in the vocabulary.

    Returns None if no tokens are in the vocabulary (edge case).
    """
    vectors = []
    for token in tokens:
        if token in model.wv:
            vectors.append(model.wv[token])
        # Out-of-vocabulary words are silently skipped

    if not vectors:
        return None  # Document has no recognizable words

    # Average all word vectors → one fixed-size document vector
    return np.mean(vectors, axis=0)


def build_document_vectors(documents: list[dict], model: Word2Vec):
    """
    Generate and save document vectors for all documents.

    Saves two numpy arrays:
      vectors.npy   — shape (N, VECTOR_SIZE), one row per document
      doc_ids.npy   — shape (N,), the document ID for each row
    """
    doc_vectors = []
    doc_ids     = []

    for doc in documents:
        combined = doc["title"] + " " + doc["content"]
        tokens   = preprocess(combined)
        vector   = document_to_vector(tokens, model)

        if vector is not None:
            doc_vectors.append(vector)
            doc_ids.append(doc["id"])
        else:
            print(f"[embeddings] WARNING: No vector for doc {doc['id']} — {doc['title']}")

    vectors_array = np.array(doc_vectors, dtype=np.float32)
    ids_array     = np.array(doc_ids,     dtype=np.int32)

    np.save(VECTORS_PATH, vectors_array)
    np.save(DOC_IDS_PATH, ids_array)

    print(f"[embeddings] Saved {len(doc_vectors)} document vectors → {VECTORS_PATH}")
    return vectors_array, ids_array


def load_document_vectors():
    """
    Load precomputed document vectors from disk.
    Returns (vectors_array, doc_ids_array).
    """
    if not os.path.exists(VECTORS_PATH) or not os.path.exists(DOC_IDS_PATH):
        raise FileNotFoundError(
            "Document vectors not found. Run: python embeddings.py  to generate them."
        )
    vectors = np.load(VECTORS_PATH)
    doc_ids = np.load(DOC_IDS_PATH)
    return vectors, doc_ids


def query_to_vector(query: str, model: Word2Vec) -> np.ndarray | None:
    """
    Preprocess a user query and convert it to a vector.
    Uses the same averaging approach as document vectors.
    """
    tokens = preprocess(query)
    return document_to_vector(tokens, model)


def initialize_embeddings(force_retrain=False):
    """
    Master setup function called at application startup.
    1. Loads or trains the Word2Vec model.
    2. Loads or generates document vectors.
    Returns (model, doc_vectors, doc_ids).
    """
    documents = get_all_documents()

    # Train or load Word2Vec model
    if force_retrain or not os.path.exists(MODEL_PATH):
        model = train_word2vec(documents)
    else:
        print("[embeddings] Loading existing Word2Vec model...")
        model = load_word2vec()

    # Build or load document vectors
    if force_retrain or not os.path.exists(VECTORS_PATH):
        doc_vectors, doc_ids = build_document_vectors(documents, model)
    else:
        print("[embeddings] Loading existing document vectors...")
        doc_vectors, doc_ids = load_document_vectors()

    print(f"[embeddings] Ready. Vectors shape: {doc_vectors.shape}")
    return model, doc_vectors, doc_ids


if __name__ == "__main__":
    # Run this file directly to train and save the model
    from document_loader import load_documents
    load_documents()
    model, vectors, ids = initialize_embeddings(force_retrain=True)

    # Quick sanity check — find most similar words
    print("\n[test] Most similar to 'machine':")
    for word, score in model.wv.most_similar("machine", topn=5):
        print(f"  {word:20s} {score:.4f}")

    print("\n[test] Most similar to 'network':")
    for word, score in model.wv.most_similar("network", topn=5):
        print(f"  {word:20s} {score:.4f}")
