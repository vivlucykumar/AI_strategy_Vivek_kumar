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

############################################################
#with hugging face 

# build_db.py
import os
import glob
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

# Use free Hugging Face MiniLM embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def build_chroma_db():
    persist_directory = "chroma_db"
    documents = []

    for file_path in glob.glob("data/*.pdf"):
        loader = PyPDFLoader(file_path)
        documents.extend(loader.load())

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    # Save to Chroma
    vectorstore = Chroma.from_documents(
        docs,
        embedding=embeddings,
        persist_directory=persist_directory,
    )
    vectorstore.persist()
    print(f"✅ Database built with {len(docs)} chunks")

if __name__ == "__main__":
    build_chroma_db()
