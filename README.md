# 🧠 RAG + QA Bot with LangChain (Open Source)

This project demonstrates how to build a **Retrieval-Augmented Generation (RAG)** pipeline 
and a **QA bot** using **LangChain**, **Chroma**, **Hugging Face embeddings**, 
and open-source LLMs.

---

## 🚀 Features
- Load PDFs with `PyPDFLoader`
- Split documents into manageable chunks
- Generate embeddings with `sentence-transformers`
- Store vectors in **Chroma DB**
- Retrieve relevant context for queries
- Answer questions with Hugging Face LLMs (e.g., Falcon, Llama 2)
- Interactive **Gradio UI** for PDF upload + query

---

## 📂 Structure
```
notebooks/      
 ├── rag_pipeline.ipynb       → step-by-step pipeline (Tasks 1–6)
 └── qa_bot_interface.ipynb   → Gradio GUI interface (Task 6 demo)
src/                        
 └── qa_bot.py                → QA bot script version
requirements.txt
README.md
LICENSE
```

---

## ▶️ Quickstart

### 1) Clone
```bash
git clone https://github.com/<your-username>/rag-qa-bot-langchain.git
cd rag-qa-bot-langchain
```

### 2) Install deps
```bash
pip install -r requirements.txt
```

---

## 📘 Option A: Run the Full Pipeline (notebook)
```bash
jupyter notebook notebooks/rag_pipeline.ipynb
```
Walkthrough:
1. Load PDF
2. Split into chunks
3. Create embeddings
4. Store in Chroma
5. Retrieve chunks
6. Ask a QA query (console output)

---

## 📘 Option B: Run the QA Bot Interface (notebook)
```bash
jupyter notebook notebooks/qa_bot_interface.ipynb
```
This launches a **Gradio** app:
1. Upload a PDF
2. Enter a query (e.g., *"What this paper is talking about?"*)
3. Read the bot's answer

When launched, Gradio shows two links:
- `http://127.0.0.1:7860` → local (works on your own machine)
- `https://xxxx.gradio.live` → public (works on cloud envs like Colab/Coursera)

Take your **QA_bot.png** screenshot from this UI.

---

## 📜 License
Free
