# 📘 Multimodal RAG Demo

This repository contains a **demo implementation of a Retrieval-Augmented Generation (RAG) pipeline** for a **single PDF**.  
It extracts **text, tables, and images** from the document, generates embeddings, stores them in a FAISS vector store, and allows you to **query the document** using natural language.

---

## 🚀 Features

- PDF ingestion (`pymupdf` + `tabula`)  
- Extracts:
  - Text chunks
  - Tables
  - Page-level images
  - Inline images
- Embedding generation (configurable: OpenAI, HuggingFace, Stub)  
- Vector store built with FAISS  
- Query support with top-k retrieval  
- Full RAG: retrieve + generate answer  

---

## 📂 Structure

```
multimodal_rag_pipeline/
├── src/
│   ├── config.py
│   ├── data_processing.py
│   ├── embedding.py
│   ├── generator.py
│   ├── rag.py
│   ├── vectorstore.py
│   └── __init__.py
├── config/
│   └── config.yaml
├── main.py
├── dynamic_rag_full_demo.ipynb
├── requirements.txt
├── Dockerfile
├── .env
└── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/<your-username>/multimodal_rag_pipeline.git
cd multimodal_rag_pipeline
pip install -r requirements.txt
```

---

## 🖥️ Usage

### CLI

```bash
# Step 1: Build FAISS index from the demo PDF
python main.py build

# Step 2: Query contexts
python main.py query "Which optimizer was used for training?"

# Step 3: Full RAG (retrieve + generate)
python main.py ask "Which optimizer was used for training?"
```

### Notebook

Run the interactive notebook:

```bash
jupyter lab dynamic_rag_full_demo.ipynb
```

---

## 🐳 Run with Docker

You can also build and run the pipeline in a container.

### Build the image

```bash
docker build -t multimodal-rag .
```

### Run Jupyter Lab inside the container

```bash
docker run -p 8888:8888 multimodal-rag
```

➡️ Then open [http://localhost:8888](http://localhost:8888) in your browser.  
By default, the container launches Jupyter Lab with no token or password.

### Run CLI inside the container

```bash
# Build index inside Docker
docker run multimodal-rag python main.py build

# Query inside Docker
docker run multimodal-rag python main.py query --text "Which optimizer was used for training?"

# Full RAG ask inside Docker
docker run multimodal-rag python main.py ask --text "Which optimizer was used for training?"
```

---

## 🔑 Notes

- This is a **demo version** → it only processes **one PDF** at a time.  
- Embeddings and generation depend on provider settings in `config/config.yaml`.  
- API keys (OpenAI, HuggingFace, AWS) should be stored in `.env`.  
- If you need secrets in Docker, pass them as environment variables:
  ```bash
  docker run -e OPENAI_API_KEY=your_key multimodal-rag python main.py ask --text "..."
  ```

---
