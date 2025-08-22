# strategy_rag.py
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

DB_PATH = "chroma_db"

# Load HuggingFace embeddings (must match build_db.py)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load Chroma vector store
vectordb = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

# Create retriever
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

# Define prompt
prompt_template = """
You are a helpful assistant for answering questions about Strategy.
Use the following context to answer the question concisely.
If the answer is not in the context, just say "I don’t know."

Context:
{context}

Question:
{question}

Answer:"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Hugging Face LLM pipeline (free, no API key)
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",   # lightweight & free
    tokenizer="google/flan-t5-base",
    max_length=512
)
llm = HuggingFacePipeline(pipeline=generator)

# Retrieval QA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=True
)
