# pip install -U langchain langchain-openai langchain-community faiss-cpu pypdf python-dotenv

# --- Educational Note: Imports and Setup ---
# We are importing various tools from LangChain:
# PyPDFLoader: Extracts text from PDF files.
# FAISS: A fast and efficient local vector database for storing embeddings.
# OpenAIEmbeddings: Converts text into continuous numerical vectors (embeddings).
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# By setting LANGCHAIN_PROJECT, we tell LangSmith to track all executions of this app under this specific project name.
os.environ['LANGCHAIN_PROJECT'] = 'RAG app'

load_dotenv()  # expects OPENAI_API_KEY in .env

PDF_PATH = "islr.pdf"  # <-- change to your PDF filename

# --- Educational Note: Step 1 - Load PDF ---
# PyPDFLoader reads the PDF and returns a list of LangChain 'Document' objects.
# By default, it creates one Document object per page of the PDF.
loader = PyPDFLoader(PDF_PATH)
docs = loader.load()

# 2) Chunk
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
splits = splitter.split_documents(docs)

# 3) Embed + index
emb = OpenAIEmbeddings(model="text-embedding-3-small")
vs = FAISS.from_documents(splits, emb)
retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# 4) Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer ONLY from the provided context. If not found, say you don't know."),
    ("human", "Question: {question}\n\nContext:\n{context}")
])

# 5) Chain
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
def format_docs(docs): return "\n\n".join(d.page_content for d in docs)

parallel = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
})

chain = parallel | prompt | llm | StrOutputParser()

# 6) Ask questions
print("PDF RAG ready. Ask a question (or Ctrl+C to exit).")
q = input("\nQ: ")
ans = chain.invoke(q.strip())
print("\nA:", ans)
