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


# ############################################################
# #with hugging face 
# strategy_rag.py
# strategy_rag.py
import os
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Patch sqlite3 with pysqlite3 if needed for Streamlit Cloud
try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

# --- 1. Embeddings (local, free, no API key needed)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# --- 2. Load Chroma vectorstore
persist_directory = "./chroma_db"
vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# --- 3. Hugging Face Hub API token
def get_hf_token():
    try:
        return st.secrets["HUGGINGFACEHUB_API_TOKEN"]
    except Exception:
        return os.getenv("HUGGINGFACEHUB_API_TOKEN")

hf_token = get_hf_token()
if not hf_token:
    raise ValueError(
        "❌ No Hugging Face API token found! "
        "Please set HUGGINGFACEHUB_API_TOKEN in .streamlit/secrets.toml or as an environment variable."
    )

# --- 4. Hugging Face LLM
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2b-it",
    max_new_tokens=512,
    temperature=0.3,
    huggingfacehub_api_token=hf_token,
)

# --- 5. Prompt template
prompt_template = """
Use the following context to answer the question at the end.
If you don’t know the answer, just say you don’t know. Be concise.

Context:
{context}

Question: {question}
Answer:
"""
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template,
)

# --- 6. Build the RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt},
)

# --- 7. CLI for local testing
if __name__ == "__main__":
    print("✅ Strategy RAG Assistant (Hugging Face client) — type 'exit' to quit.\n")
    while True:
        query = input("Question: ")
        if query.lower() in ["exit", "quit"]:
            break
        result = qa_chain.invoke({"query": query})
        response = result.get("result", "⚠️ No answer found.")
        print("\nAnswer:", response, "\n")