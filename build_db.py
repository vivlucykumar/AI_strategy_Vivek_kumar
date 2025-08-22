# build_db.py
import os
from langchain_community.document_loaders import PyPDFDirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

DATA_PATH = "data"
DB_PATH = "chroma_db"

def build_vectorstore():
    # Load PDFs and text files
    loaders = [
        PyPDFDirectoryLoader(DATA_PATH),
        TextLoader(os.path.join(DATA_PATH, "extra.txt")) if os.path.exists(os.path.join(DATA_PATH, "extra.txt")) else None
    ]
    documents = []
    for loader in loaders:
        if loader:
            documents.extend(loader.load())

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    # HuggingFace embeddings (no API key needed)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create Chroma vector store
    vectordb = Chroma.from_documents(docs, embeddings, persist_directory=DB_PATH)
    vectordb.persist()
    print(f"✅ Chroma DB created at: {DB_PATH} with {len(docs)} chunks.")

if __name__ == "__main__":
    build_vectorstore()
