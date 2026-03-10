# pip install -U langchain langchain-openai langchain-community faiss-cpu pypdf python-dotenv langsmith

import os
from dotenv import load_dotenv

from langsmith import traceable  # <-- key import

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# --- LangSmith env (make sure these are set) ---
# LANGCHAIN_TRACING_V2=true
# LANGCHAIN_API_KEY=...
# LANGCHAIN_PROJECT=pdf_rag_demo

load_dotenv()

PDF_PATH = "islr.pdf"  # change to your file

# ---------- Educational Note: Traced Setup Steps ----------
# The @traceable decorator automatically logs the inputs, outputs, and execution time of these functions to LangSmith.
# We also attach 'tags' and 'metadata' to help filter or analyze these specific setup steps later in the LangSmith UI.

# This function loads the PDF. Tagged as 'setup' with metadata specifying the loader used.
@traceable(name="load_pdf", tags=["setup"], metadata={'loader': 'PyPDFLoader'})
def load_pdf(path: str):
    loader = PyPDFLoader(path)
    return loader.load()  # list[Document]

# This function splits the loaded documents into smaller chunks.
@traceable(name="split_documents", tags=["setup"], metadata={'splitter': 'RecursiveCharacterTextSplitter'})
def split_documents(docs, chunk_size=1000, chunk_overlap=150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)

# This function takes the text chunks, embeds them via OpenAI, and indexes them in a FAISS vector database.
@traceable(name="build_vectorstore", tags=["setup"], metadata={'vectorstore': 'FAISS'})
def build_vectorstore(splits):
    emb = OpenAIEmbeddings(model="text-embedding-3-small")
    # FAISS.from_documents internally calls the embedding model:
    vs = FAISS.from_documents(splits, emb)
    return vs

# --- Educational Note: Umbrella Pipeline ---
# You can also trace an entire “setup” umbrella span.
# Because this function calls other @traceable functions, LangSmith will display a nested trace hierarchy!
@traceable(name="setup_pipeline")
def setup_pipeline(pdf_path: str):
    docs = load_pdf(pdf_path)
    splits = split_documents(docs)
    vs = build_vectorstore(splits)
    return vs

# ---------- Educational Note: Pipeline Creation ----------
# The 'llm' is the heart of the generative part of RAG. We use a low temperature for more factual, deterministic answers.
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# The 'prompt' enforces that the bot only uses the retrieved context to answer the human's question.
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer ONLY from the provided context. If not found, say you don't know."),
    ("human", "Question: {question}\n\nContext:\n{context}")
])

# 'format_docs' is a simple helper function taking the retrieved Document objects 
# and joining their raw page_content into a single large string for the prompt.
def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

# Build the index under traced setup
vectorstore = setup_pipeline(PDF_PATH)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

parallel = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough(),
})

chain = parallel | prompt | llm | StrOutputParser()

# ---------- run a query (also traced) ----------
print("PDF RAG ready. Ask a question (or Ctrl+C to exit).")
q = input("\nQ: ").strip()

# Give the visible run name + tags/metadata so it’s easy to find:
config = {
    "run_name": "pdf_rag_query"
}

ans = chain.invoke(q, config=config)
print("\nA:", ans)
