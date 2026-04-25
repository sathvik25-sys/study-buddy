import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEndpointEmbeddings  # ✅ correct import

load_dotenv()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_embeddings():
    return HuggingFaceEndpointEmbeddings(  # ✅ correct class name
        model="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=os.getenv("HF_API_KEY")  # ✅ correct param name
    )

def create_vector_store(text):
    print("✂️ Splitting text into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = splitter.split_text(text)
    print(f"✅ Created {len(chunks)} chunks")

    print("🔢 Creating embeddings via HuggingFace API (no local model)...")
    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=get_embeddings(),
        persist_directory="./chroma_db"
    )
    print(f"✅ Done! {len(chunks)} chunks stored in ChromaDB")
    return vectorstore


def load_vector_store():
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=get_embeddings()
    )
    return vectorstore