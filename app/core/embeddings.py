"""
app/core/embeddings.py

Utility functions for generating and comparing text embeddings.
We use the lightweight 'all-MiniLM-L6-v2' model for fast, local semantic similarity.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

# Load the model once globally so it's cached in memory.
# all-MiniLM-L6-v2 is small (~80MB) and fast, perfect for CPU inference.
# It outputs 384-dimensional vectors.
MODEL_NAME = "all-MiniLM-L6-v2"
_model = None

def get_model() -> SentenceTransformer:
    """Lazy load the model to avoid slow startup times if not used."""
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model

def embed(text: str) -> np.ndarray:
    """
    Generate a 384-dimensional embedding vector for the given text.
    The vectors are L2-normalized so we can use dot product for cosine similarity.
    """
    model = get_model()
    # encode() returns a numpy array. We normalize it.
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calculate the cosine similarity between two normalized embedding vectors.
    Since they are normalized, cosine similarity is just the dot product.
    Returns a float between -1.0 and 1.0 (though practically 0.0 to 1.0 for this model).
    """
    # Ensure they are 1D arrays
    a = np.asarray(a).flatten()
    b = np.asarray(b).flatten()
    
    # Dot product of normalized vectors == cosine similarity
    score = np.dot(a, b)
    
    # Clip to avoid float precision issues outside [-1.0, 1.0]
    return float(np.clip(score, -1.0, 1.0))
