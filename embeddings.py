import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

load_dotenv()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def create_vector_store(text):
    print("✂️ Splitting text into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = splitter.split_text(text)
    print(f"✅ Created {len(chunks)} chunks")

    print("🔢 Creating embeddings via HuggingFace API (no local model)...")
    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=os.getenv("HF_API_KEY"),
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("🗄️ Storing in ChromaDB...")
    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    print(f"✅ Done! {len(chunks)} chunks stored in ChromaDB")
    return vectorstore


def load_vector_store():
    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=os.getenv("HF_API_KEY"),
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    return vectorstore