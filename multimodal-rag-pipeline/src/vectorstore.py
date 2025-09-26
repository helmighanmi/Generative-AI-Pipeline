"""
Author: GHANMI Helmi
Date: 2025-09-26
Position: Data-Science
"""

import faiss, numpy as np, json, os

class FaissVectorStore:
    def __init__(self, index_path, metadata_path):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.index = None
        self.metadata = []

    def build(self, embeddings, items):
        emb = np.array([e for e in embeddings if e is not None]).astype("float32")
        dim = emb.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(emb)
        self.metadata = items

    def save(self):
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path,"w") as f: json.dump(self.metadata,f)

    def load(self):
        self.index = faiss.read_index(self.index_path)
        with open(self.metadata_path) as f: self.metadata = json.load(f)

    def search(self, query_emb, top_k=5):
        if self.index is None: raise RuntimeError("Index not loaded")
        q = np.array([query_emb]).astype("float32")
        D,I = self.index.search(q, top_k)
        return [self.metadata[i] for i in I[0] if i < len(self.metadata)]
