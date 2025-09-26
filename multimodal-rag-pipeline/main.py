"""
Author: GHANMI Helmi
Date: 2025-09-26
Position: Data-Science
"""

import argparse

from src.config import Config
from src.embedding import EmbeddingService
from src.vectorstore import FaissVectorStore
from src.rag import retrieve, rag_ask
from src.data_processing import download_pdf, create_directories, process_text_chunks, process_tables, process_images, process_page_images

import pymupdf
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter


def build_index():
    config = Config()
    paths = config.get_data_paths()
    pdf_url = "https://arxiv.org/pdf/1706.03762.pdf"
    filename = "attention.pdf"

    filepath = download_pdf(pdf_url, paths["input_dir"], filename)
    create_directories(paths["output_dir"])

    doc = pymupdf.open(filepath)
    items = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.get_pipeline_config()["chunk_size"],
        chunk_overlap=config.get_pipeline_config()["chunk_overlap"]
    )

    for page_num, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            process_text_chunks(filepath, text, splitter, page_num, paths["output_dir"], items)
        process_tables(filepath, doc, page_num, paths["output_dir"], items)
        process_images(doc, page, page_num, paths["output_dir"], items)
        process_page_images(page, page_num, paths["output_dir"], items)

    # Embedding
    embedder = EmbeddingService()
    embeddings = []
    with tqdm(total=len(items), desc="Embedding items") as bar:
        for it in items:
            if it["type"] == "text":
                emb = embedder.embed(text=it["text"])
            elif it["type"] in ["image", "page"]:
                emb = embedder.embed(image_b64=it["image"])
            else:
                emb = None
            it["embedding"] = emb
            embeddings.append(emb)
            bar.update(1)

    vs_cfg = config.get_vectorstore_config()
    store = FaissVectorStore(index_path=vs_cfg["index_path"], metadata_path=vs_cfg["metadata_path"])
    store.build(embeddings, items)
    store.save()
    print("✅ Index built and saved")


def query_index(query: str):
    config = Config()
    vs_cfg = config.get_vectorstore_config()
    store = FaissVectorStore(index_path=vs_cfg["index_path"], metadata_path=vs_cfg["metadata_path"])
    store.load()

    results = retrieve(store, query, top_k=config.get_retriever_config()["top_k"])
    for r in results:
        print(f"Page {r['page']} ({r['type']})")
        if "text" in r:
            print(r["text"][:200])
        print("---")


def ask_question(question: str):
    config = Config()
    vs_cfg = config.get_vectorstore_config()
    store = FaissVectorStore(index_path=vs_cfg["index_path"], metadata_path=vs_cfg["metadata_path"])
    store.load()

    answer = rag_ask(store, question, top_k=config.get_retriever_config()["top_k"])
    print("🤖 Answer: ", answer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multimodal RAG Pipeline")
    parser.add_argument("command", choices=["build", "query", "ask"], help="Pipeline command")
    parser.add_argument("--text", type=str, help="Query or question text")
    args = parser.parse_args()

    if args.command == "build":
        build_index()
    elif args.command == "query":
        if not args.text:
            raise ValueError("Please provide --text for query")
        query_index(args.text)
    elif args.command == "ask":
        if not args.text:
            raise ValueError("Please provide --text for ask")
        ask_question(args.text)
