import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()


def create_vector_store(text):
    print("✂️ Splitting text into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = splitter.split_text(text)
    print(f"✅ Created {len(chunks)} chunks")

    print("🔢 Creating embeddings (HuggingFace - free)...")
    embeddings = HuggingFaceEmbeddings(
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
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L2-v2"
    )
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    return vectorstore


if __name__ == "__main__":
    from pdf_loader import extract_text_from_pdf

    text = extract_text_from_pdf("sample.pdf")
    print(f"📄 Extracted {len(text)} characters from PDF")

    vectorstore = create_vector_store(text)
    print("🎉 Vector store created successfully!")