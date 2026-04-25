import os
import time
import base64
import tempfile
import streamlit as st
from elevenlabs.client import ElevenLabs
from duckduckgo_search import DDGS
from pdf_loader import extract_text_from_pdf
from embeddings import create_vector_store, load_vector_store
from rag_chain import create_rag_chain, ask_question
from quiz import generate_quiz
from image_extractor import extract_images_from_pdf
from exam_mode import generate_exam_qa


def ask_with_retry(chain, question, retries=3, delay=30):
    for i in range(retries):
        try:
            return ask_question(chain, question)
        except Exception as e:
            if "RESOURCE_EXHAUSTED" in str(e):
                st.warning(f"⏳ Rate limit hit. Retrying in {delay}s... ({i+1}/{retries})")
                time.sleep(delay)
            else:
                raise e
    raise Exception("Max retries reached. Please wait a few minutes.")


def text_to_audio(text):
    client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
    audio = client.text_to_speech.convert(
        voice_id="JBFqnCBsd6RMkjVDRZzb",
        text=text,
        model_id="eleven_multilingual_v2"
    )
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
        for chunk in audio:
            f.write(chunk)
        return f.name


def play_audio(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    st.markdown(f"""
        <audio controls autoplay>
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
    """, unsafe_allow_html=True)
    os.remove(file_path)


def get_topic_image(question: str):
    try:
        with DDGS(headers={"User-Agent": "Mozilla/5.0"}) as ddgs:
            results = list(ddgs.images(
                keywords=question,
                max_results=3,
                type_image="photo"
            ))
            if results:
                return results[0]["image"]
        return None
    except Exception:
        return None


st.set_page_config(
    page_title="Study Buddy",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Study Buddy")
st.caption("Powered by Groq + RAG")

# ── Session state init ────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "rag_chain" not in st.session_state:
    st.session_state["rag_chain"] = None
if "doc_processed" not in st.session_state:
    st.session_state["doc_processed"] = False
if "play_audio_index" not in st.session_state:
    st.session_state["play_audio_index"] = None
if "quiz_questions" not in st.session_state:
    st.session_state["quiz_questions"] = []
if "quiz_answers" not in st.session_state:
    st.session_state["quiz_answers"] = {}
if "quiz_submitted" not in st.session_state:
    st.session_state["quiz_submitted"] = False
if "pdf_text" not in st.session_state:
    st.session_state["pdf_text"] = ""
if "pdf_images" not in st.session_state:
    st.session_state["pdf_images"] = []
if "exam_qa" not in st.session_state:
    st.session_state["exam_qa"] = []

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📂 Upload Your Document")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())

        if st.button("🔄 Process Document"):
            with st.spinner("Reading PDF..."):
                text = extract_text_from_pdf("temp.pdf")
                st.session_state["pdf_text"] = text

            with st.spinner("Extracting images..."):
                st.session_state["pdf_images"] = extract_images_from_pdf("temp.pdf")

            with st.spinner("Creating embeddings..."):
                vs = create_vector_store(text)

            with st.spinner("Setting up RAG chain..."):
                st.session_state["rag_chain"] = create_rag_chain(vs)
                st.session_state["doc_processed"] = True
                st.session_state["messages"] = []
                st.session_state["quiz_questions"] = []
                st.session_state["quiz_submitted"] = False
                st.session_state["exam_qa"] = []

            st.success("✅ Document processed! Ask your questions.")

    if st.button("📦 Load Saved Notes"):
        with st.spinner("Loading..."):
            try:
                vs = load_vector_store()
                st.session_state["rag_chain"] = create_rag_chain(vs)
                st.session_state["doc_processed"] = True
                st.success("✅ Loaded saved notes!")
            except Exception as e:
                st.error(f"❌ {e}")

    if st.session_state["doc_processed"]:
        st.success("🟢 Ready to answer")
    else:
        st.warning("⚠️ No document loaded")

    if st.button("🗑️ Clear Chat"):
        st.session_state["messages"] = []
        st.session_state["play_audio_index"] = None
        st.rerun()

    st.divider()

    if st.button("📝 Generate Quiz"):
        if not st.session_state["pdf_text"]:
            st.warning("⚠️ Please process a document first.")
        else:
            with st.spinner("Generating quiz..."):
                try:
                    st.session_state["quiz_questions"] = generate_quiz(st.session_state["pdf_text"])
                    st.session_state["quiz_answers"] = {}
                    st.session_state["quiz_submitted"] = False
                    st.success("✅ Quiz ready!")
                except Exception as e:
                    st.error(f"❌ Failed to generate quiz: {e}")

    if st.button("🎓 Exam Mode"):
        if not st.session_state["pdf_text"]:
            st.warning("⚠️ Please process a document first.")
        else:
            with st.spinner("Generating exam questions..."):
                try:
                    st.session_state["exam_qa"] = generate_exam_qa(st.session_state["pdf_text"])
                    st.success("✅ Exam questions ready!")
                except Exception as e:
                    st.error(f"❌ Failed: {e}")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "📝 Quiz", "🖼️ Images", "🎓 Exam Mode"])

# ── Chat Tab ──────────────────────────────────────────────────────────────────
with tab1:
    st.header("💬 Ask Questions About Your Document")

    for i, msg in enumerate(st.session_state["messages"]):
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("sources") and msg["sources"] != "N/A":
                st.caption(f"📎 Sources: {msg['sources']}")
            if msg.get("image_url"):
                st.image(msg["image_url"], caption="📸 Related Image", width=300)

            if msg["role"] == "assistant":
                if st.button("🔊 Play Audio", key=f"audio_{i}"):
                    st.session_state["play_audio_index"] = i
                if st.session_state["play_audio_index"] == i:
                    with st.spinner("Generating audio..."):
                        audio_file = text_to_audio(msg["content"])
                        play_audio(audio_file)
                    st.session_state["play_audio_index"] = None

    question = st.chat_input("Ask anything about your document...")

    if question:
        if not st.session_state["doc_processed"] or st.session_state["rag_chain"] is None:
            st.warning("⚠️ Please upload and process a PDF first.")
            st.stop()

        st.session_state["messages"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = ask_with_retry(st.session_state["rag_chain"], question)
                    st.write(result["answer"])
                    if result["sources"] != "N/A":
                        st.caption(f"📎 Sources: {result['sources']}")

                    # 🖼️ Fetch related image from DuckDuckGo
                    img_url = get_topic_image(question)
                    if img_url:
                        st.image(img_url, caption=f"📸 Related: {question[:40]}", width=300)

                    st.session_state["messages"].append({
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result.get("sources", "N/A"),
                        "image_url": img_url
                    })
                except Exception as e:
                    st.error(f"❌ Error: {e}")

# ── Quiz Tab ──────────────────────────────────────────────────────────────────
with tab2:
    st.header("📝 Quiz Yourself")

    if not st.session_state["quiz_questions"]:
        st.info("👈 Click 'Generate Quiz' in the sidebar to start!")
    else:
        with st.form("quiz_form"):
            for i, q in enumerate(st.session_state["quiz_questions"]):
                st.subheader(f"Q{i+1}: {q['question']}")
                st.session_state["quiz_answers"][i] = st.radio(
                    label=f"Select answer for Q{i+1}",
                    options=q["options"],
                    key=f"q_{i}",
                    label_visibility="collapsed"
                )
                st.divider()

            submitted = st.form_submit_button("✅ Submit Quiz")

        if submitted:
            st.session_state["quiz_submitted"] = True

        if st.session_state["quiz_submitted"]:
            score = 0
            st.header("📊 Results")
            for i, q in enumerate(st.session_state["quiz_questions"]):
                user_ans = st.session_state["quiz_answers"].get(i)
                correct = q["answer"]
                if user_ans == correct:
                    score += 1
                    st.success(f"✅ Q{i+1}: Correct! — {correct}")
                else:
                    st.error(f"❌ Q{i+1}: Wrong! Your answer: {user_ans} | Correct: {correct}")

            st.subheader(f"🎯 Score: {score} / {len(st.session_state['quiz_questions'])}")

            if score == len(st.session_state["quiz_questions"]):
                st.balloons()
                st.success("🏆 Perfect score! Amazing!")
            elif score >= 3:
                st.info("👍 Good job! Keep studying!")
            else:
                st.warning("📚 Keep practicing!")

            if st.button("🔄 Retake Quiz"):
                st.session_state["quiz_answers"] = {}
                st.session_state["quiz_submitted"] = False
                st.rerun()

# ── Images Tab ────────────────────────────────────────────────────────────────
with tab3:
    st.header("🖼️ Images from Document")

    if not st.session_state["pdf_images"]:
        st.info("👈 Process a document to extract images!")
    else:
        st.success(f"Found {len(st.session_state['pdf_images'])} image(s)")
        cols = st.columns(3)
        for idx, img in enumerate(st.session_state["pdf_images"]):
            with cols[idx % 3]:
                st.image(
                    f"data:image/{img['ext']};base64,{img['b64']}",
                    caption=f"Page {img['page']} - Image {img['index']}",
                    use_container_width=True
                )

# ── Exam Mode Tab ─────────────────────────────────────────────────────────────
with tab4:
    st.header("🎓 Exam Mode — Important Questions & Answers")

    if not st.session_state["exam_qa"]:
        st.info("👈 Click 'Exam Mode' in the sidebar to generate questions!")
    else:
        st.success(f"✅ {len(st.session_state['exam_qa'])} important questions generated!")
        for i, qa in enumerate(st.session_state["exam_qa"]):
            with st.expander(f"Q{i+1}: {qa['question']}"):
                st.markdown(f"**Answer:** {qa['answer']}")