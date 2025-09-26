"""
Author: GHANMI Helmi
Date: 2025-09-26
Position: Data-Science
"""

from src.embedding import EmbeddingService
from src.vectorstore import FaissVectorStore
from src.generator import GeneratorService

def retrieve(store:FaissVectorStore, query:str, top_k:int=5):
    emb = EmbeddingService().embed(text=query)
    return store.search(emb, top_k=top_k)

def rag_ask(store:FaissVectorStore, question:str, top_k:int=5) -> str:
    ctx = retrieve(store, question, top_k=top_k)
    return GeneratorService().generate(question, ctx)
