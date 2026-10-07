"""
preprocessing.py — Text cleaning pipeline.

Steps applied to both documents and queries:
  1. Lowercase
  2. Remove punctuation and digits
  3. Tokenize (split on whitespace)
  4. Remove stop words
  5. Lemmatize each token

The output is a list of clean tokens ready for Word2Vec.
"""

import re
import nltk

# Download required NLTK data on first run (silent after that)
for resource in ["stopwords", "wordnet", "punkt", "punkt_tab", "omw-1.4"]:
    try:
        nltk.data.find(f"corpora/{resource}" if resource not in ("punkt", "punkt_tab") else f"tokenizers/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

# Regex: keep only letters and spaces (remove punctuation, digits, symbols)
CLEAN_PATTERN = re.compile(r"[^a-z\s]")


def preprocess(text: str) -> list[str]:
    """
    Clean and tokenize a string.

    Parameters
    ----------
    text : str
        Raw text (document content or user query).

    Returns
    -------
    list[str]
        List of clean, lemmatized tokens.
    """
    # Step 1: lowercase
    text = text.lower()

    # Step 2: remove everything except letters and spaces
    text = CLEAN_PATTERN.sub(" ", text)

    # Step 3: split into tokens
    tokens = text.split()

    # Step 4 & 5: remove stop words and lemmatize in one pass
    clean_tokens = [
        LEMMATIZER.lemmatize(token)
        for token in tokens
        if token not in STOP_WORDS and len(token) > 2
    ]

    return clean_tokens


def preprocess_for_display(text: str) -> str:
    """
    Return the cleaned tokens joined as a single string.
    Useful for debugging — shows what the model actually sees.
    """
    return " ".join(preprocess(text))


if __name__ == "__main__":
    sample = "How can doctors identify and diagnose diseases using machine learning?"
    tokens = preprocess(sample)
    print("Input:", sample)
    print("Tokens:", tokens)
