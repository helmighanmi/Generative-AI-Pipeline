"""
Author: GHNAMI Helmi
Date: 2025-09-26
Position: Data-Science
"""

import faiss
import json
import numpy as np
import os

class FaissVectorStore:
    def __init__(self, index_path="data/faiss_index.index", metadata_path="data/faiss_metadata.json"):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.index = None
        self.items = []

    def build(self, embeddings, items):
        dim = len(embeddings[0])
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(embeddings).astype(np.float32))
        self.items = items

    def save(self):
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, "w") as f:
            json.dump(self.items, f)

    def load(self):
        self.index = faiss.read_index(self.index_path)
        with open(self.metadata_path, "r") as f:
            self.items = json.load(f)

    def search(self, query_vector, top_k=5):
        D, I = self.index.search(np.array(query_vector).reshape(1, -1).astype(np.float32), top_k)
        return [self.items[i] for i in I[0]]
