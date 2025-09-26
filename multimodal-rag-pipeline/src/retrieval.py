"""
Author: GHNAMI Helmi
Date: 2025-09-26
Position: Data-Science
"""

import numpy as np
import faiss

class FaissRetriever:
    def __init__(self, dimension: int):
        """
        Initialize a FAISS index with L2 (Euclidean) distance.

        Args:
            dimension (int): Dimension of the embedding vectors.
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.index.reset()

    def add_embeddings(self, embeddings):
        """
        Add embeddings to the FAISS index.

        Args:
            embeddings (List[np.ndarray] or np.ndarray): List or array of embedding vectors.
        """
        np_embeddings = np.array(embeddings, dtype=np.float32)
        self.index.add(np_embeddings)

    def search(self, query_embedding, top_k=5):
        """
        Search the FAISS index for nearest neighbors of the query embedding.

        Args:
            query_embedding (np.ndarray): The query embedding vector.
            top_k (int): Number of nearest neighbors to return.

        Returns:
            Tuple[np.ndarray, np.ndarray]: distances and indices of nearest neighbors.
        """
        query_vector = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        distances, indices = self.index.search(query_vector, top_k)
        return distances, indices
