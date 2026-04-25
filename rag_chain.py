import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


def create_rag_chain(vectorstore):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set in your .env file.")

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=api_key,
        temperature=0.2
    )

    system_prompt = (
        "You are a helpful academic assistant called Study Buddy.\n"
        "Use the following retrieved context to answer the question accurately.\n"
        "If the answer is not in the context, say: 'I don't have enough information to answer that.'\n"
        "Keep your answer concise and well-structured.\n\n"
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)

    rag_chain = create_retrieval_chain(
        vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        ),
        question_answer_chain
    )

    return rag_chain


def ask_question(chain, question):
    if not question or not question.strip():
        return {
            "answer": "Please provide a valid question.",
            "sources": "N/A"
        }

    result = chain.invoke({"input": question})

    context_docs = result.get("context", [])
    sources = []

    if context_docs:
        for doc in context_docs:
            source = doc.metadata.get("source", None)
            page = doc.metadata.get("page", None)
            if source:
                label = f"{os.path.basename(source)}"
                if page is not None:
                    label += f" (p.{page + 1})"
                sources.append(label)

    source_text = ", ".join(sorted(set(sources))) if sources else "Retrieved chunks"

    return {
        "answer": result.get("answer") or "I could not generate an answer.",
        "sources": source_text
    }


if __name__ == "__main__":
    from embeddings import load_vector_store

    try:
        vs = load_vector_store()
        chain = create_rag_chain(vs)

        print("🤖 Study Buddy is ready!")

        while True:
            question = input("\n📚 Ask a question (or type 'exit' to quit): ").strip()
            if question.lower() in ("exit", "quit"):
                print("👋 Goodbye!")
                break

            res = ask_question(chain, question)
            print(f"\n🧠 Buddy: {res['answer']}")
            print(f"📄 Sources: {res['sources']}")

    except ValueError as ve:
        print(f"⚙️  Config Error: {ve}")
    except Exception as e:
        print(f"❌ Error: {e}")