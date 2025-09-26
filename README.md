# 🤖 Generative AI Pipeline

This repository is a collection of **Generative AI experiments and pipelines**, with a focus on **RAG (Retrieval-Augmented Generation)** and **LangChain**.  
It is organized into multiple sub-projects, each exploring a different approach.

---

## 📂 Repository Structure

```
Generative-AI-Pipeline/
├── langchain/                  # LangChain examples and utilities
├── multimodal-rag-pipeline/    # Demo: multimodal RAG (PDFs with text, tables, images)
├── rag-qa-bot-langchain/       # Q&A bot powered by LangChain + RAG
└── README.md                   # This file
```

---

## 🚀 Sub-Projects

### 🔹 `langchain/`
Experiments and examples using the [LangChain](https://www.langchain.com/) framework.  
Covers document loaders, chains, prompt templates, and basic RAG workflows.

---

### 🔹 `multimodal-rag-pipeline/`
A **demo RAG pipeline** that ingests a single PDF, extracts **text, tables, and images**, generates embeddings, stores them in **FAISS**, and allows you to **query and generate answers**.  

- PDF ingestion: `pymupdf`, `tabula`  
- Embeddings: HuggingFace / OpenAI / Stub (configurable)  
- Vector store: FAISS  
- Demo notebook included  

📖 See the [README here](multimodal-rag-pipeline/README.md).

---

### 🔹 `rag-qa-bot-langchain/`
A **RAG-powered Q&A chatbot** built with **LangChain**.  
Supports conversational interaction over indexed documents.

- Uses LangChain document loaders and vector stores  
- Embeddings: OpenAI / HuggingFace  
- LLMs: OpenAI GPT models or local HuggingFace models  

---

## 🛠️ Installation

Clone the repo:

```bash
git clone https://github.com/helmighanmi/Generative-AI-Pipeline.git
cd Generative-AI-Pipeline
```

Each sub-project has its own `requirements.txt`.  
For example, to install the multimodal RAG demo:

```bash
cd multimodal-rag-pipeline
pip install -r requirements.txt
```

---

## 🐳 Docker

Some sub-projects include Dockerfiles.  
Example for the multimodal RAG pipeline:

```bash
cd multimodal-rag-pipeline
docker build -t multimodal-rag .
docker run -p 8888:8888 multimodal-rag
```

---

## 🔑 Environment Variables

Create a `.env` file at the project root or inside each sub-project with:

```bash
OPENAI_API_KEY=your_openai_api_key
HUGGINGFACE_HUB_TOKEN=your_hf_token
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
AWS_DEFAULT_REGION= # us-east-1 or else
```

---

## 👤 Author

**Helmi Ghanmi**  
Data Scientist
📅 2025-09-26

---
