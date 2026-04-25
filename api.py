import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
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

chain = None  # loaded lazily

class QuestionRequest(BaseModel):
    question: str

class QuizRequest(BaseModel):
    text: str

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/ask")
def ask(req: QuestionRequest):
    global chain
    if chain is None:
        from embeddings import load_vector_store
        vs = load_vector_store()
        chain = create_rag_chain(vs)
    result = ask_question(chain, req.question)
    return result

@app.post("/quiz")
def quiz(req: QuizRequest):
    questions = generate_quiz(req.text)
    return {"questions": questions}