import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpointEmbeddings

load_dotenv()

def get_embeddings():
    return HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=os.getenv("HF_API_KEY")
    )

def create_vector_store(text):
    print("Splitting text into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = splitter.split_text(text)
    print(f"Created {len(chunks)} chunks")
    print("Creating embeddings via HuggingFace API...")
    vectorstore = FAISS.from_texts(
        texts=chunks,
        embedding=get_embeddings()
    )
    vectorstore.save_local("./faiss_db")
    print(f"Done! {len(chunks)} chunks stored")
    return vectorstore

def load_vector_store():
    return FAISS.load_local(
        "./faiss_db",
        get_embeddings(),
        allow_dangerous_deserialization=True
    )
