from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
import sys
# Patch sqlite3 with pysqlite3
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
import ollama

# -------------------------
# 1. Use Ollama client directly
# -------------------------
def ollama_llm(prompt: str) -> str:
    response = ollama.generate(model="llama3", prompt=prompt)
    return response["response"]

llm = RunnableLambda(lambda x: ollama_llm(x.to_string()))

# -------------------------
# 2. Embeddings (must match build_db.py)
# -------------------------
embedding_function = OllamaEmbeddings(model="nomic-embed-text")

# -------------------------
# 3. Load Chroma vectorstore
# -------------------------
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding_function,
)
retriever = vectorstore.as_retriever()

# -------------------------
# 4. Define the RAG prompt
# -------------------------
prompt = PromptTemplate(
    template="""
You are a helpful assistant for answering questions about business strategy.

Use the following document context to answer the question.
If the answer cannot be found in the context, say you don’t know.

Context:
{context}

Question:
{input}

Answer:
""",
    input_variables=["context", "input"],
)

# -------------------------
# 5. Create chain
# -------------------------
document_chain = create_stuff_documents_chain(llm, prompt)
qa_chain = create_retrieval_chain(retriever, document_chain)

# -------------------------
# 6. Interactive Q&A loop
# -------------------------
if __name__ == "__main__":
    print("✅ Strategy RAG Assistant (Ollama client) — type 'exit' to quit.\n")
    while True:
        query = input("Question: ")
        if query.lower() in ["exit", "quit"]:
            break
        result = qa_chain.invoke({"input": query})
        print("\nAnswer:", result["answer"], "\n")
