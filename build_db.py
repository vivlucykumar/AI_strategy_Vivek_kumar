import os
import glob
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# -------------------------
# CONFIG
# -------------------------
PDF_DIR = r"C:\Vivek\Personal\Documents\IIMA Strategic mgt course\16_Business Ideas\AI_strategy\data"
DB_DIR = "./chroma_db"       # Where the vector DB will be stored
EMBED_MODEL = "nomic-embed-text"  # ✅ Embedding model (must pull it in Ollama first)

# -------------------------
# INIT EMBEDDINGS + SPLITTER
# -------------------------
embedding_function = OllamaEmbeddings(model=EMBED_MODEL)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

# -------------------------
# CLEAN OLD DB (optional)
# -------------------------
# if os.path.exists(DB_DIR):
#     print(f"🧹 Removing old Chroma DB at {DB_DIR} ...")
#     shutil.rmtree(DB_DIR)

# -------------------------
# LOAD PDF FILES
# -------------------------
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

# -------------------------
# SAVE INTO CHROMA VECTORSTORE
# -------------------------
if all_docs:
    vectorstore = Chroma.from_documents(
        documents=all_docs,
        embedding=embedding_function,
        persist_directory=DB_DIR,
    )
    # vectorstore.persist()   # ❌ Not needed in langchain-chroma
    print(f"\n🎉 Successfully built Chroma DB with {len(all_docs)} chunks.")
    print(f"📂 Stored at: {DB_DIR}")
else:
    print("\n⚠️ No documents were added to the database.")

