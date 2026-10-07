"""
document_loader.py — Reads documents.json and populates SQLite.
Run this once (or whenever the dataset changes).
"""

import json
import os
from database import create_tables, insert_document, document_count, clear_documents

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "documents.json")


def load_documents(force_reload=False):
    """
    Load documents from JSON into SQLite.
    If documents already exist and force_reload is False, skip loading.
    Returns the number of documents in the database after loading.
    """
    create_tables()

    if document_count() > 0 and not force_reload:
        print(f"[loader] Database already has {document_count()} documents. Skipping reload.")
        return document_count()

    if force_reload:
        clear_documents()
        print("[loader] Cleared existing documents for reload.")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at: {DATA_PATH}")

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        documents = json.load(f)

    for doc in documents:
        insert_document(
            doc_id=doc["id"],
            title=doc["title"],
            category=doc["category"],
            content=doc["content"],
            file_name=doc.get("file_name", ""),
        )

    count = document_count()
    print(f"[loader] Loaded {count} documents into SQLite.")
    return count


if __name__ == "__main__":
    load_documents(force_reload=True)
