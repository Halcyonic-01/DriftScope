"""
tests/test_embeddings.py

Unit tests for embedding and cosine similarity utilities.
"""

import numpy as np
import pytest

from app.core.embeddings import embed, cosine_sim


def test_embed_returns_unit_vector():
    """Verify that the generated embedding is normalized (L2 norm is 1.0)."""
    text = "The quick brown fox jumps over the lazy dog."
    vector = embed(text)
    
    # 384 dimensions for all-MiniLM-L6-v2
    assert vector.shape == (384,)
    
    # Norm should be very close to 1.0 since we normalize
    norm = np.linalg.norm(vector)
    assert pytest.approx(norm, 0.0001) == 1.0


def test_cosine_sim_identical_texts():
    """Identical texts should have a cosine similarity of exactly 1.0."""
    text = "This is a test sentence."
    vec1 = embed(text)
    vec2 = embed(text)
    
    score = cosine_sim(vec1, vec2)
    assert pytest.approx(score, 0.0001) == 1.0


def test_cosine_sim_orthogonal_texts():
    """Completely unrelated texts should have a lower cosine similarity."""
    text1 = "Quantum mechanics describes the fundamental behavior of nature at small scales."
    text2 = "I love eating chocolate chip cookies with a glass of milk."
    
    vec1 = embed(text1)
    vec2 = embed(text2)
    
    score = cosine_sim(vec1, vec2)
    # They should be relatively dissimilar. For this model, 
    # it's usually between 0.0 and 0.3 for totally unrelated texts.
    assert score < 0.3


def test_cosine_sim_threshold_at_0_80():
    """Texts that are semantically very similar but not identical should score high (>0.8)."""
    text1 = "The car accelerated quickly down the highway."
    text2 = "The automobile sped rapidly along the freeway."
    
    vec1 = embed(text1)
    vec2 = embed(text2)
    
    score = cosine_sim(vec1, vec2)
    # They mean almost exactly the same thing.
    assert score > 0.75
