# Semantic Search Engine Using Word Embeddings

A college-level **Semantic Search Engine** that retrieves documents based on the **meaning and context of a user's query**, rather than relying only on exact keyword matching.

The project uses **Word2Vec-based word embeddings** to represent text as numerical vectors and compares the semantic similarity between the user's query and indexed documents.

---

## 📌 Project Overview

Traditional search engines mainly depend on keyword matching. This can fail when the query and document use different words but have similar meanings.

For example:

> **Query:** `How can I learn programming?`

A traditional keyword search may prioritize documents containing the exact words "learn" and "programming".

A semantic search engine can also retrieve documents such as:

> "A beginner's guide to software development and coding."

because the concepts are semantically related.

This project demonstrates how **Natural Language Processing (NLP), Word Embeddings, Vector Similarity, and Information Retrieval** can be combined to build a basic semantic search system.

---

## ✨ Features

* 🔎 Semantic document search
* 🧠 Word2Vec-based word embeddings
* 📝 Text preprocessing
* 📚 Document loading and indexing
* 📊 Vector-based document representation
* 📐 Similarity-based ranking
* 🎯 Rocchio relevance feedback
* 📈 Search evaluation
* 💾 SQLite-based document/search storage
* 🌐 Simple web-based frontend
* ⚡ Python backend API
* 📂 Local document dataset support

---

## 🏗️ Project Architecture

```text
                    ┌──────────────────────┐
                    │       User           │
                    │   Search Query       │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │      Frontend        │
                    │ HTML + CSS + JS      │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │    Python Backend    │
                    │       Flask API      │
                    └──────────┬───────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
      ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
      │ Preprocessing│ │  Word2Vec    │ │   Database   │
      │    Module    │ │  Embeddings  │ │   / Storage  │
      └──────┬───────┘ └──────┬───────┘ └──────────────┘
             │                 │
             └────────┬────────┘
                      ▼
              ┌───────────────┐
              │ Vector Search │
              │ & Ranking     │
              └───────┬───────┘
                      │
                      ▼
              ┌───────────────┐
              │ Search Results │
              └───────────────┘
```

---

## 📁 Project Structure

```text
semantic-search-engine/
│
├── backend/
│   ├── app.py
│   ├── database.py
│   ├── document_loader.py
│   ├── embeddings.py
│   ├── evaluation.py
│   ├── preprocessing.py
│   ├── rocchio.py
│   ├── search.py
│   ├── requirements.txt
│   │
│   └── data/
│       ├── documents.json
│       ├── doc_ids.npy
│       ├── vectors.npy
│       ├── search.db
│       └── w2v_model/
│
├── frontend/
│   ├── index.html
│   ├── script.js
│   └── style.css
│
└── README.md
```

---

## 🛠️ Technologies Used

### Backend

* Python
* Flask
* NumPy
* SQLite
* Gensim / Word2Vec
* NLP preprocessing
* Vector similarity

### Frontend

* HTML5
* CSS3
* JavaScript

### Machine Learning / NLP

* Word Embeddings
* Word2Vec
* Text preprocessing
* Vector representation
* Semantic similarity
* Information Retrieval
* Rocchio relevance feedback

---

## 🔄 How the System Works

### 1. Document Collection

The system receives a collection of documents that are stored and processed for indexing.

### 2. Text Preprocessing

Documents are cleaned and prepared before generating embeddings.

Typical preprocessing includes:

```text
Raw Text
   ↓
Lowercase Conversion
   ↓
Tokenization
   ↓
Cleaning
   ↓
Stopword Handling
   ↓
Processed Text
```

### 3. Word Embedding

The processed words are converted into numerical vectors using **Word2Vec**.

Conceptually:

```text
"computer" → [0.21, -0.14, 0.53, ...]
"software" → [0.18, -0.11, 0.49, ...]
```

Words with similar meanings tend to have similar vector representations.

### 4. Document Vector Generation

Word vectors are combined to create a vector representation of each document.

```text
Document
   ↓
Words
   ↓
Word Embeddings
   ↓
Document Vector
```

### 5. Query Processing

When a user enters a search query, the same preprocessing and embedding process is applied to the query.

### 6. Similarity Calculation

The query vector is compared with document vectors.

The system ranks documents according to their semantic similarity to the query.

### 7. Result Ranking

The most relevant documents are returned to the user.

---

## 🎯 Rocchio Relevance Feedback

The project also includes a **Rocchio relevance feedback** module.

Rocchio is an Information Retrieval technique that modifies a query vector based on relevant and non-relevant documents.

Conceptually:

```text
Original Query
      ↓
Search Results
      ↓
User Feedback
      ↓
Relevant / Non-Relevant Documents
      ↓
Updated Query Representation
      ↓
Improved Search Results
```

This allows the search system to improve retrieval based on relevance information.

---

## 📊 Evaluation

The project includes an evaluation module for measuring search performance.

Evaluation can be used to analyze:

* Precision
* Recall
* F-measure
* Ranking quality
* Search relevance

These metrics help determine how effectively the system retrieves relevant documents.

---

# 🚀 Installation

## 1. Clone the Repository

```bash
git clone https://github.com/shivam-techstack/semantic-search-engine.git
```

Move into the project directory:

```bash
cd semantic-search-engine
```

---

## 2. Create a Python Virtual Environment

Navigate to the backend:

```bash
cd backend
```

Create a virtual environment:

### Windows

```powershell
python -m venv venv
```

Activate it:

```powershell
venv\Scripts\activate
```

If PowerShell blocks script execution, you can use:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then:

```powershell
venv\Scripts\activate
```

---

## 3. Install Dependencies

Run:

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Backend

From the `backend` directory:

```bash
python app.py
```

The Flask backend should start on the configured local address.

For example:

```text
http://127.0.0.1:5000
```

Check the terminal output for the exact port used by the application.

---

# 🌐 Running the Frontend

The frontend is located in:

```text
frontend/
```

You can open:

```text
frontend/index.html
```

directly in a browser, or serve the frontend using a local development server.

For VS Code, the **Live Server** extension can be used.

The frontend communicates with the backend API to submit search queries and display results.

---

# 🔍 Example Search

Example query:

```text
machine learning algorithms
```

The system processes the query and retrieves documents that are semantically related to the concept, even if they do not contain exactly the same keywords.

Example flow:

```text
User Query
    ↓
Text Preprocessing
    ↓
Word2Vec Embedding
    ↓
Query Vector
    ↓
Similarity Calculation
    ↓
Document Ranking
    ↓
Relevant Results
```

---

# 📚 Main Modules

| Module               | Purpose                                 |
| -------------------- | --------------------------------------- |
| `app.py`             | Backend API and application entry point |
| `database.py`        | Database operations                     |
| `document_loader.py` | Loading and managing documents          |
| `preprocessing.py`   | Text preprocessing                      |
| `embeddings.py`      | Word embedding generation               |
| `search.py`          | Semantic search and ranking             |
| `rocchio.py`         | Relevance feedback                      |
| `evaluation.py`      | Search performance evaluation           |

---

# 🧠 Key Concepts Demonstrated

This project demonstrates practical implementation of:

* Natural Language Processing
* Information Retrieval
* Word Embeddings
* Word2Vec
* Vector Space Representation
* Semantic Similarity
* Document Ranking
* Relevance Feedback
* Search Evaluation
* REST API
* Frontend–Backend Integration

---

# 🎓 Academic Purpose

This project is designed as a **college-level Information Storage and Retrieval / NLP project**.

It demonstrates how semantic search can be implemented using traditional NLP and machine-learning techniques without depending on large commercial search APIs.

### Learning Objectives

By developing this project, students can understand:

1. How search engines process text.
2. How documents can be represented as vectors.
3. How Word2Vec generates word embeddings.
4. How semantic similarity can improve search.
5. How documents can be ranked according to relevance.
6. How Rocchio relevance feedback works.
7. How search systems can be evaluated.
8. How a Python backend can communicate with a web frontend.

---

# 🔮 Future Improvements

Possible future enhancements include:

* Transformer-based embeddings
* Sentence-BERT (SBERT)
* FAISS vector search
* Larger document datasets
* Advanced ranking algorithms
* Improved relevance feedback
* User authentication
* Search history
* Search analytics
* PDF/document upload
* Multilingual semantic search
* Cloud deployment
* REST API documentation

---

# ⚠️ Current Scope

This project focuses on demonstrating **semantic search using Word2Vec and vector-based information retrieval**.

It is intentionally designed to remain understandable and suitable for an academic project rather than implementing a production-scale search engine.

---

# 👨‍💻 Author

**Shivam Kumar**

GitHub:
https://github.com/shivam-techstack

---

# 📄 License

This project is intended for **educational and academic purposes**.

You may modify and extend the project for learning and experimentation.
