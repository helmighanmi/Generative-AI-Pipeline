from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA

def build_qa_bot(pdf_path):
    # Load PDF
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    # Split text
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(pages)

    # Embeddings
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Chroma DB
    chroma_db = Chroma.from_documents(docs, embedding)
    retriever = chroma_db.as_retriever(search_kwargs={"k": 3})

    # LLM (requires HF Hub token: huggingface-cli login)
    llm = HuggingFaceHub(
        repo_id="tiiuae/falcon-7b-instruct",
        model_kwargs={"temperature": 0.5, "max_new_tokens": 256}
    )

    # QA bot
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

if __name__ == "__main__":
    pdf_path = "heat_bath_paper.pdf"  # replace with your PDF
    qa_bot = build_qa_bot(pdf_path)
    query = "What this paper is talking about?"
    result = qa_bot({"query": query})
    print("\n--- QA Bot Answer ---")
    print(result["result"])
