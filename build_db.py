# import os
# import glob
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_ollama import OllamaEmbeddings
# from langchain_community.vectorstores import Chroma
# import chromadb

# # -------------------------
# # CONFIG
# # -------------------------
# PDF_DIR = r"C:\Vivek\Personal\Documents\IIMA Strategic mgt course\16_Business Ideas\AI_strategy\data"
# DB_DIR = "./chroma_db"       # Where the vector DB will be stored
# EMBED_MODEL = "nomic-embed-text"  # ✅ Embedding model (must pull it in Ollama first)
# COLLECTION_NAME = "strategy_docs" # Chroma collection name

# # -------------------------
# # INIT EMBEDDINGS + SPLITTER
# # -------------------------
# embedding_function = OllamaEmbeddings(model=EMBED_MODEL)

# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000,
#     chunk_overlap=200,
# )

# # -------------------------
# # LOAD PDF FILES
# # -------------------------
# all_docs = []
# pdf_files = glob.glob(os.path.join(PDF_DIR, "*.pdf"))
# if not pdf_files:
#     print(f"⚠️ No PDF files found in {PDF_DIR}")
# else:
#     for pdf_file in pdf_files:
#         try:
#             loader = PyPDFLoader(pdf_file)
#             docs = loader.load()
#             splits = text_splitter.split_documents(docs)
#             all_docs.extend(splits)
#             print(f"✅ Loaded {len(splits)} chunks from {os.path.basename(pdf_file)}")
#         except Exception as e:
#             if "password" in str(e).lower() or "encrypted" in str(e).lower():
#                 print(f"⏭️ Skipping password-protected file: {os.path.basename(pdf_file)}")
#             else:
#                 print(f"❌ Error loading {pdf_file}: {e}")

# # -------------------------
# # SAVE INTO CHROMA VECTORSTORE
# # -------------------------
# if all_docs:
#     client = chromadb.PersistentClient(path=DB_DIR)
#     vectorstore = Chroma.from_documents(
#         documents=all_docs,
#         embedding=embedding_function,
#         client=client,
#         collection_name=COLLECTION_NAME,
#     )
#     print(f"\n🎉 Successfully built Chroma DB with {len(all_docs)} chunks.")
#     print(f"📂 Stored at: {DB_DIR}")
# else:
#     print("\n⚠️ No documents were added to the database.")

# # ############################################################
# build_db.py

import os
import glob
import streamlit as st # Only needed for accessing secrets in a local run
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import chromadb
import sys
import asyncio

# --- FIX for RuntimeError: no running event loop ---
try:
    loop = asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
# ---------------------------------------------------

# -------------------------
# CONFIG
# -------------------------
# Define the directory where your PDF files are stored.
PDF_DIR = "./data"
# Define the directory for the vector database.
DB_DIR = "./chroma_db"
# Name of the Chroma collection.
COLLECTION_NAME = "strategy_docs"

# -------------------------
# INIT EMBEDDINGS + SPLITTER
# -------------------------
# We use Google's embedding function and get the API key from Streamlit secrets
def get_google_api_key():
    """Retrieves Google API key from Streamlit secrets or environment variables."""
    try:
        return st.secrets["GOOGLE_API_KEY"]
    except Exception:
        return os.getenv("GOOGLE_API_KEY")

google_api_key = get_google_api_key()
if not google_api_key:
    print("❌ Google API key not found. Please set 'GOOGLE_API_KEY'.")
    sys.exit(1)

embedding_function = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=google_api_key,
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

def build_vector_store():
    """Builds the Chroma vector store from PDF documents."""
    print("Starting vector store build process...")
    all_docs = []
    pdf_files = glob.glob(os.path.join(PDF_DIR, "*.pdf"))
    if not pdf_files:
        print(f"⚠️ No PDF files found in {PDF_DIR}")
    else:
        for pdf_file in pdf_files:
            try:
                loader = PyPDFLoader(pdf_file)
                docs = loader.load()
                splits = text_splitter.split_documents(docs)
                all_docs.extend(splits)
                print(f"✅ Loaded {len(splits)} chunks from {os.path.basename(pdf_file)}")
            except Exception as e:
                if "password" in str(e).lower() or "encrypted" in str(e).lower():
                    print(f"⏭️ Skipping password-protected file: {os.path.basename(pdf_file)}")
                else:
                    print(f"❌ Error loading {pdf_file}: {e}")

    if all_docs:
        client = chromadb.PersistentClient(path=DB_DIR)
        vectorstore = Chroma.from_documents(
            documents=all_docs,
            embedding=embedding_function,
            client=client,
            collection_name=COLLECTION_NAME,
        )
        print(f"\n🎉 Successfully built Chroma DB with {len(all_docs)} chunks.")
        print(f"📂 Stored at: {DB_DIR}")
    else:
        print("\n⚠️ No documents were added to the database.")

if __name__ == "__main__":
    build_vector_store()
