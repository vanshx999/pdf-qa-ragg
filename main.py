import os 
import shutil
import numpy as np
import tempfile
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.prompts import  ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from rank_bm25 import BM25Okapi
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse


app = FastAPI(title='RAG API - Any PDF')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],  # Allow all origins (for development only)
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

PERSIST_DIR = './chroma_fresh'
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
TOP_K = 3

vectorstore = None
bm25 = None
chunks = []
tokenized_chunks = []

def tokenize(text):
    return text.lower().split()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not set")


llm =  ChatGroq(
    model='llama-3.3-70b-versatile',
    temperature=0.0,
    groq_api_key=os.getenv('GROQ_API_KEY')
)  

prompt = ChatPromptTemplate.from_template("""
Answer based only on the following context.
Cite sources as [Source: Page X].

Context:
{context}

Question: {question}

Answer:
""")

def build_indexes(pdf_path: str):
    global vectorstore, bm25, chunks, tokenized_chunks

    print(f'Building indexfor: {pdf_path}')

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len  # Character count, not tokens
    )
    chunks = splitter.split_documents(documents)
    print(f'Spliting into{len(chunks)} chunks')

    shutil.rmtree(PERSIST_DIR, ignore_errors=True)
    embeddings = HuggingFaceBgeEmbeddings(model_name=MODEL_NAME)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
    )

    tokenized_chunks = [tokenize(doc.page_content) for doc in chunks]
    bm25 = BM25Okapi(tokenized_chunks)

    print('Index ready.')
    return len(chunks)

def hybrid_retriever(query, k=3):
    if not vectorstore or not bm25:
        raise ValueError('No PDF uploaded yet, PDF /upload first. ')

    scores = {}
    k_rrf = 60

    vector_docs = vectorstore.similarity_search(query, k=k)
    for rank, doc in enumerate(vector_docs):
        try:
            idx = chunks.index(doc)
            scores[idx] = scores.get(idx, 0) + 1 / (k_rrf + rank + 1)
        except ValueError:
            pass

    tokenized_query = tokenize(query)
    bm25_scores =  bm25.get_scores(tokenized_query)
    bm25_top = np.argsort(bm25_scores)[::1][:k]
    for rank, idx in enumerate(bm25_top):
        scores[idx] = scores.get(idx, 0) + 1 / (k_rrf + rank + 1)

    sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
    return [chunks[idx] for idx, _ in sorted_results]
 
def format_docs(docs):
    return "\n\n".join(f'[Page {doc.metadata.get('page', '?')}]: {doc.page_content[:400]}' for doc in docs)

class Question(BaseModel):
    question: str

class Answer(BaseModel):
    answer: str
    sources: list[dict]

class UploadResponse(BaseModel):
    message: str
    chunks_created: int

@app.get("/", response_class=HTMLResponse)
def home():
    with open("index.html", "r") as f:
        return f.read()
        
@app.post('/upload', response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """Upload any PDF. Replaces current index."""
    
    # Validate
    if not file.filename.endswith('.pdf'):
        return {'message': 'Only PDF files allowed', 'chunks_created': 0}
    
    # Save to temp file
    suffix = f'_{file.filename}'
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        # Build indexes
        count = build_indexes(tmp_path)
        return {
            'message': f'Uploaded and indexed {file.filename}',
            'chunks_created': count
        }
    finally:
        os.unlink(tmp_path)  # Clean up temp file

@app.post('/ask', response_model=Answer)
def ask(question: Question):
    if not vectorstore:
        return {
            'answer': 'No PDF uploaded yet. POST /upload first.',
            'sources': []
        }
    
    docs = hybrid_retriever(question.question, k=TOP_K)
    context = format_docs(docs)
    
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({'context': context, 'question': question.question})
    
    sources = [
        {
            'page': doc.metadata.get('page', '?'),
            'text': doc.page_content[:200]
        }
        for doc in docs
    ]
    
    return {'answer': answer, 'sources': sources}

@app.get('/health')
def health():
    return {
        'status': 'ok',
        'pdf_loaded': vectorstore is not None,
        'chunks_loaded': len(chunks)
    }
@app.get("/upload")
def upload_form():
    return {"message": "Use POST with multipart/form-data to upload a PDF file"}
