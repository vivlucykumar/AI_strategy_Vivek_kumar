# strategy_rag.py
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI  # You can switch this to Hugging Face LLM if needed

DB_PATH = "chroma_db"

# Load HuggingFace embeddings (must match build_db.py)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load Chroma vector store
vectordb = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

# Create retriever
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

# Define prompt
prompt_template = """
You are an assistant for answering questions about Strategy.
Use the provided context to answer the question.
If the answer is not in the context, say "I don’t know."

Context:
{context}

Question:
{question}

Answer:"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Use OpenAI (requires key) OR swap with local HuggingFace pipeline
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Retrieval QA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=True
)
    