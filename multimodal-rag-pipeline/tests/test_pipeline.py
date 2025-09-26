"""
Author: GHANMI Helmi
Date: 2025-09-26
Position: Data-Science
"""

import numpy as np
import pytest

from src.embedding import EmbeddingService
from src.generator import GeneratorService


def test_embedding_stub_text(monkeypatch):
    """
    Test that text embeddings in stub mode
    return a deterministic vector of the right shape.
    """
    service = EmbeddingService()
    service.provider = "stub"
    service.dim = 128  # override for test
    vec1 = service.embed(text="hello world")
    vec2 = service.embed(text="hello world")

    assert isinstance(vec1, np.ndarray)
    assert vec1.shape == (128,)
    # Deterministic → same input should yield identical vectors
    assert np.allclose(vec1, vec2)


def test_embedding_stub_image(monkeypatch):
    """
    Test that image embeddings in stub mode
    produce a vector of the configured dimension.
    """
    service = EmbeddingService()
    service.provider = "stub"
    service.dim = 64  # override for test
    fake_img = "aGVsbG8="  # base64("hello")

    vec = service.embed(image_b64=fake_img)
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (64,)


def test_generator_stub():
    """
    Test that GeneratorService in stub mode
    returns a string containing the question and context.
    """
    gen = GeneratorService()
    gen.provider = "stub"

    question = "What is attention?"
    contexts = [{"text": "Attention is all you need."}]
    answer = gen.generate(question, contexts)

    assert isinstance(answer, str)
    assert "STUB ANSWER" in answer
    assert "Attention is all you need." in answer
