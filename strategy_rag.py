# # Patch sqlite3 with pysqlite3 (needed for Streamlit Cloud / Python 3.13)
# import sys
# try:
#     __import__("pysqlite3")
#     sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
# except ImportError:
#     pass

# import chromadb
# import ollama
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import PromptTemplate
# from langchain_core.runnables import RunnableLambda
# from langchain_community.vectorstores import Chroma
# from langchain_ollama import OllamaEmbeddings

# # -------------------------
# # 1. LLM via Ollama
# # -------------------------
# def ollama_llm(prompt: str) -> str:
#     response = ollama.generate(model="llama3", prompt=prompt)
#     return response["response"]

# llm = RunnableLambda(lambda x: ollama_llm(x.to_string()))

# # -------------------------
# # 2. Embeddings
# # -------------------------
# embedding_function = OllamaEmbeddings(model="nomic-embed-text")

# # -------------------------
# # 3. Load Chroma vectorstore
# # -------------------------
# DB_DIR = "./chroma_db"
# COLLECTION_NAME = "strategy_docs"

# client = chromadb.PersistentClient(path=DB_DIR)
# vectorstore = Chroma(
#     client=client,
#     collection_name=COLLECTION_NAME,
#     embedding_function=embedding_function,
# )
# retriever = vectorstore.as_retriever()

# # -------------------------
# # 4. Define the RAG prompt
# # -------------------------
# prompt = PromptTemplate(
#     template="""You are a helpful assistant for answering questions about business strategy.

# Use the following document context to answer the question.
# If the answer cannot be found in the context, say you don’t know.

# Context:
# {context}

# Question:
# {input}

# Answer:""",
#     input_variables=["context", "input"],
# )

# # -------------------------
# # 5. Create chain
# # -------------------------
# document_chain = create_stuff_documents_chain(llm, prompt)
# qa_chain = create_retrieval_chain(retriever, document_chain)

# # -------------------------
# # 6. CLI for local testing
# # -------------------------
# if __name__ == "__main__":
#     print("✅ Strategy RAG Assistant (Ollama client) — type 'exit' to quit.\n")
#     while True:
#         query = input("Question: ")
#         if query.lower() in ["exit", "quit"]:
#             break
#         result = qa_chain.invoke({"input": query})
#         print("\nAnswer:", result["answer"], "\n")


# # ############################################################
# Patch sqlite3 with pysqlite3 (needed for Streamlit Cloud / Python 3.13)
# import sys
# try:
#     __import__("pysqlite3")
#     sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
# except ImportError:
#     pass

# import os
# import chromadb
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import PromptTemplate
# from langchain_community.vectorstores import Chroma
# # Patch sqlite3 with pysqlite3 (needed for Streamlit Cloud / Python 3.13)
# import sys
# try:
#     __import__("pysqlite3")
#     sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
# except ImportError:
#     pass

# import asyncio

# # --- FIX for RuntimeError: no running event loop ---
# # This code creates and sets a new event loop if one is not already running.
# try:
#     loop = asyncio.get_running_loop()
# except RuntimeError:
#     loop = asyncio.new_event_loop()
#     asyncio.set_event_loop(loop)
# # ---------------------------------------------------

# import os
# import chromadb
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import PromptTemplate
# from langchain_community.vectorstores import Chroma
# from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# # New imports for Google GenAI
# from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# # -------------------------
# # 1. LLM via Google GenAI (Gemini)
# # -------------------------
# # Get API key from Streamlit secrets
# llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", google_api_key=os.environ.get("GOOGLE_API_KEY"))

# # -------------------------
# # 2. Embeddings
# # -------------------------
# embedding_function = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=os.environ.get("GOOGLE_API_KEY"))

# # -------------------------
# # 3. Load Chroma vectorstore
# # -------------------------
# DB_DIR = "./chroma_db"
# COLLECTION_NAME = "strategy_docs"

# client = chromadb.PersistentClient(path=DB_DIR)
# vectorstore = Chroma(
#     client=client,
#     collection_name=COLLECTION_NAME,
#     embedding_function=embedding_function,
# )
# retriever = vectorstore.as_retriever()

# # -------------------------
# # 4. Define the RAG prompt
# # -------------------------
# prompt = PromptTemplate(
#     template="""You are a helpful assistant for answering questions about business strategy.

# Use the following document context to answer the question.
# If the answer cannot be found in the context, say you don’t know.

# Context:
# {context}

# Question:
# {input}

# Answer:""",
#     input_variables=["context", "input"],
# )

# # -------------------------
# # 5. Create chain
# # -------------------------
# document_chain = create_stuff_documents_chain(llm, prompt)
# qa_chain = create_retrieval_chain(retriever, document_chain)

# # -------------------------
# # 6. CLI for local testing
# # -------------------------
# if __name__ == "__main__":
#     print("✅ Strategy RAG Assistant (Gemini) — type 'exit' to quit.\n")
#     while True:
#         query = input("Question: ")
#         if query.lower() in ["exit", "quit"]:
#             break
#         result = qa_chain.invoke({"input": query})
#         print("\nAnswer:", result["answer"], "\n")

###########################################################################################################
# Patch sqlite3 with pysqlite3 (needed for Streamlit Cloud / Python 3.13)
import os
import sys
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import chromadb
import ollama
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# -------------------------
# 1. LLM via Google GenAI (Gemini)
# -------------------------
# Get API key from Streamlit secrets
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", google_api_key=os.environ.get("GOOGLE_API_KEY"))

# -------------------------
# 2. Embeddings
# -------------------------
embedding_function = OllamaEmbeddings(model="nomic-embed-text")

# -------------------------
# 3. Load Chroma vectorstore
# -------------------------
DB_DIR = "./chroma_db"
COLLECTION_NAME = "strategy_docs"

client = chromadb.PersistentClient(path=DB_DIR)
vectorstore = Chroma(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding_function=embedding_function,
)
retriever = vectorstore.as_retriever()

# -------------------------
# 4. Define the RAG prompt
# -------------------------
prompt = PromptTemplate(
    template="""You are a helpful assistant for answering questions about business strategy.

Use the following document context to answer the question.
If the answer cannot be found in the context, say you don’t know.

Context:
{context}

Question:
{input}

Answer:""",
    input_variables=["context", "input"],
)

# -------------------------
# 5. Create chain
# -------------------------
document_chain = create_stuff_documents_chain(llm, prompt)
qa_chain = create_retrieval_chain(retriever, document_chain)

# -------------------------
# 6. CLI for local testing
# -------------------------
if __name__ == "__main__":
    print("✅ Strategy RAG Assistant (Ollama client) — type 'exit' to quit.\n")
    while True:
        query = input("Question: ")
        if query.lower() in ["exit", "quit"]:
            break
        result = qa_chain.invoke({"input": query})
        print("\nAnswer:", result["answer"], "\n")