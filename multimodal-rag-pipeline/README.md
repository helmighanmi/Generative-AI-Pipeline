# multimodal-rag-pipeline

This project implements a Retrieval-Augmented Generation (RAG) pipeline using multimodal inputs (text, tables, images) extracted from PDFs.  
It leverages Amazon Bedrock models (Titan for embeddings, Nova for generation) and FAISS for vector similarity search.

## Features

- PDF data extraction (text, tables, images, page snapshots)
- Amazon Titan Multimodal Embeddings for unified embedding generation
- FAISS vector store for similarity search
- Amazon Nova model for RAG question answering with multimodal context
- Configurable via YAML file
- Dockerized environment
- CI/CD pipeline via GitHub Actions

## Setup

1. Clone the repo:

```bash
git clone https://github.com/yourusername/multimodal-rag-pipeline.git
cd multimodal-rag-pipeline

Install dependencies:

pip install -r requirements.txt


Configure AWS credentials for boto3 (needed for Amazon Bedrock access).

Run the demo notebook:

jupyter notebook notebooks/demo_multimodal_rag.ipynb

Usage

Modify the config file in config/config.yaml to customize the pipeline, input directories, and model parameters.

Docker

Build and run Docker container:

docker build -t multimodal-rag-pipeline .
docker run -p 8888:8888 multimodal-rag-pipeline


Access Jupyter notebook at http://localhost:8888