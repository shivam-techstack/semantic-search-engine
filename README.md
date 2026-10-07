# Semantic Search Engine v2 — Using Word Embeddings

An Information Storage and Retrieval (ISR) college project demonstrating **4 retrieval modes**
inspired by the [google-but-smarter](https://github.com/amar21-ai/google-but-smarter) project.

---

## What's New in v2

| Feature | Description |
|---|---|
| 🌐 Query Expansion | Adds Word2Vec similar words to query before searching |
| 🎯 Rocchio Feedback | Moves query vector toward top results and re-ranks |
| 📋 Inverted Index | Term lookup — see posting lists for any word |
| 🔗 More Like This | Find documents similar to any result |
| ⚖️ 4-Mode Compare | Side-by-side comparison of all 4 retrieval modes |

---

## Quick Start

```bash
cd backend
pip install -r requirements.txt
python app.py
```

Open **http://127.0.0.1:5000**

> First run trains Word2Vec (~15–30 seconds). Every run after that starts in under 2 seconds.

---

## Technology Stack

| Component | Technology |
|---|---|
| Backend / API | Python 3.11+, Flask |
| Embeddings | Gensim Word2Vec (Skip-gram, 100-dim) |
| Text Processing | NLTK (tokenize, stopwords, lemmatize) |
| Vector Math | NumPy |
| Similarity | scikit-learn cosine_similarity |
| Keyword Search | scikit-learn TfidfVectorizer |
| Database | SQLite (built-in Python) |
| Frontend | HTML5, CSS3, Vanilla JavaScript |

---

## Project Structure

```
semantic-search-engine/
│
├── backend/
│   ├── app.py              ← Flask server — 10 API routes
│   ├── search.py           ← Semantic, Keyword, Expanded, Compare
│   ├── rocchio.py          ← Rocchio relevance feedback algorithm  ← NEW
│   ├── embeddings.py       ← Word2Vec training & document vectors
│   ├── preprocessing.py    ← Text cleaning pipeline
│   ├── document_loader.py  ← JSON → SQLite loader
│   ├── database.py         ← SQLite helpers
│   ├── evaluation.py       ← Precision, Recall, F1 metrics
│   ├── requirements.txt
│   └── data/
│       ├── documents.json  ← 80 sample documents (9 categories)
│       ├── search.db       ← SQLite (auto-generated)
│       ├── vectors.npy     ← Document vectors (auto-generated)
│       ├── doc_ids.npy     ← ID-to-vector mapping (auto-generated)
│       └── w2v_model/      ← Trained Word2Vec model (auto-generated)
│
├── frontend/
│   ├── index.html          ← 5-tab UI
│   ├── style.css
│   └── script.js
│
└── README.md
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/search?q=&type=&k=` | Search — type: semantic / keyword / expanded / rocchio |
| GET | `/api/compare?q=&k=` | Semantic vs keyword side-by-side |
| GET | `/api/compare/all?q=&k=` | All 4 modes side-by-side |
| GET | `/api/similar/<id>?k=` | More-Like-This for a document |
| GET | `/api/index/lookup?word=` | Inverted index posting list |
| GET | `/api/evaluate?k=` | Precision / Recall / F1 |
| GET | `/api/documents` | List all documents |
| GET | `/api/document/<id>` | Single document details |
| GET | `/api/status` | Engine health check |

---

## The 4 Retrieval Modes Explained

### 1. Semantic Search (Word2Vec)
```
Query → preprocess → average Word2Vec vectors → cosine similarity vs all docs → rank
```
Understands meaning. Finds "AI in Healthcare" for query "doctors detect diseases"
even with no word overlap.

### 2. Keyword Search (TF-IDF)
```
Query → preprocess → TF-IDF vector → cosine similarity vs TF-IDF matrix → rank
```
Classic IR. Only matches documents containing the exact (preprocessed) query words.

### 3. Query Expansion (Word2Vec)
```
Query → preprocess → find similar words in W2V vocab → expand token list
     → average expanded vectors → cosine similarity → rank
```
Example: "disease detection" expands to "disease detection diagnosis medical identify symptom"
before searching. Bridges the vocabulary gap.

### 4. Rocchio Relevance Feedback
```
Query → initial semantic search (top-3) → centroid of top-3 vectors
      → new_query = 1.0 × original + 0.75 × centroid → normalize
      → re-rank all documents with improved query vector
```
Named after Gerard Salton (1971). Automatically improves the query without user input.

---

## Inverted Index

At startup, the engine builds an inverted index mapping every preprocessed token
to the sorted list of document IDs that contain it:

```
"neural"    → [5, 13, 14, 15, 16, 17, 57, 61, 75]
"encrypt"   → [27, 28, 35]
"cluster"   → [8, 24, 66]
```

Use the **Inverted Index** tab to look up any word and see its posting list.

---

## More Like This

Every search result has a **"More like this"** button. It uses the document's own
precomputed vector as the query and returns the most semantically similar documents
in the collection — same as the feature in google-but-smarter.

---

## Evaluation Results (K=5, 10 test queries)

| Metric | Semantic (Word2Vec) | Keyword (TF-IDF) |
|---|---|---|
| Precision@5 | 52.0% | 48.0% |
| Recall@5 | 51.7% | 47.5% |
| F1-Score | 51.4% | 47.4% |

Semantic search outperforms TF-IDF on all metrics. The advantage is largest for
queries where the vocabulary differs from document content.

---

## Sample Queries to Try

```
How can doctors detect diseases?
securing computer networks from hackers
machine learning classification algorithms
storing and processing large amounts of data
neural networks for image recognition
text search and document retrieval
predicting future values from historical data
privacy and data protection
how does internet communication work
generative models for creating content
```

---

## Concepts Covered (for Viva)

1. Document collection and metadata storage (SQLite)
2. Text preprocessing pipeline (NLTK)
3. Word2Vec Skip-gram architecture
4. Document vector generation (averaged word vectors)
5. TF-IDF term weighting
6. Inverted index and posting lists
7. Cosine similarity (vector space model)
8. Query expansion using word embeddings
9. Rocchio relevance feedback algorithm
10. Top-K retrieval and result ranking
11. More-Like-This document similarity
12. Precision@K, Recall@K, F1-score evaluation

---

## Limitations & Future Scope

**Limitations:**
- Word2Vec trained on small corpus (80 docs) — larger pre-trained model would generalise better
- Averaged vectors lose word order
- No user session or query history

**Future Scope:**
- Pre-trained GloVe / FastText vectors
- TF-IDF weighted averaging instead of simple mean
- Sentence-BERT for sentence-level embeddings
- User-provided relevance feedback (true Rocchio, not pseudo)
- BM25 as additional keyword baseline
