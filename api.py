import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import FAISS
from rag_chain import create_rag_chain, ask_question
from quiz import generate_quiz

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

chain = None

def get_embeddings():
    return HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=os.getenv("HF_API_KEY")
    )

class QuestionRequest(BaseModel):
    question: str

class TextRequest(BaseModel):
    text: str

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/load")
def load_text(req: TextRequest):
    global chain
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_text(req.text)
    vs = FAISS.from_texts(texts=chunks, embedding=get_embeddings())
    chain = create_rag_chain(vs)
    return {"status": "loaded", "chunks": len(chunks)}

@app.post("/ask")
def ask(req: QuestionRequest):
    global chain
    if chain is None:
        return {"answer": "No document loaded. Please upload a PDF first.", "sources": "N/A"}
    result = ask_question(chain, req.question)
    return result

@app.post("/quiz")
def quiz(req: TextRequest):
    questions = generate_quiz(req.text)
    return {"questions": questions}