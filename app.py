import streamlit as st
import fitz  # PyMuPDF
from bckend import load_content, build_faiss_index, ask_question_streaming, smart_chunk_text

st.set_page_config(page_title="WebBrain – Your Website/PDF Answer Bot", layout="wide")

st.markdown(
    """
    <h1 style='text-align: center; margin-bottom: 7px;'>🧠 WebBrain</h1>
    <h4 style='text-align: center; color: gray;'>Ask questions and get instant answers from any website or PDF</h4>
    """,
    unsafe_allow_html=True
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "index" not in st.session_state:
    st.session_state.index = None
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "url" not in st.session_state:
    st.session_state.url = ""

# --- Input Type Selection ---
input_type = st.radio("Choose input type:", ["Website URL", "PDF Document"])

pdf_file = None
url = ""

if input_type == "Website URL":
    url = st.text_input("Enter Website URL", value=st.session_state.get("url", ""), placeholder="https://example.com")
    if url:
        st.caption(f"Preview of: {url}")
        st.components.v1.iframe(url, height=500)
else:
    pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])

load_clicked = st.button("Load Content")

question_type = st.selectbox(
    "❓ What kind of question will you ask?",
    ["Open-ended", "Close-ended (Yes/No)", "Fact-based", "Definition"]
)

# --- PDF Text Extraction Helper ---
def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# --- Load and Index Content ---
if load_clicked:
    with st.spinner("🔄 Loading and indexing content..."):
        try:
            if input_type == "Website URL" and url:
                chunks = load_content(url, depth=2)
                st.session_state.url = url
            elif input_type == "PDF Document" and pdf_file:
                text = extract_text_from_pdf(pdf_file)
                if not text.strip():
                    raise ValueError("❌ No text found in the uploaded PDF.")
                chunks = smart_chunk_text(text)
            else:
                raise ValueError("❗ Please provide a valid URL or upload a PDF.")

            index, chunk_list = build_faiss_index(chunks)
            st.session_state.index = index
            st.session_state.chunks = chunk_list
            st.session_state.messages = []
            st.success("✅ Content loaded and indexed successfully!")

        except Exception as e:
            st.session_state.index = None
            st.session_state.chunks = []
            st.error(f"❌ Failed to load content: {e}")

# --- Chat History Display ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Chat Input ---
if prompt := st.chat_input("Ask a question about the content..."):
    if st.session_state.index is None:
        st.warning("⚠️ Please load a website or PDF first before asking questions.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ""
            for token in ask_question_streaming(
                prompt,
                st.session_state.index,
                st.session_state.chunks,
                k=5,
                question_type=question_type,
            ):
                full_response += token
                placeholder.markdown(full_response + "▌")

            placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
