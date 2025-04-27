import os
import re
import streamlit as st
import streamlit.components.v1 as components
from datetime import datetime
from werkzeug.utils import secure_filename

from src.gpp import GPP, GPPConfig
from src.qa import AnswerGenerator

class ContextAwareAnswerGenerator:
    def __init__(self, chunks):
        self.chunks = chunks
        self.original_generator = AnswerGenerator(chunks)

    def answer(self, question, conversation_context=None):
        if not conversation_context or len(conversation_context) <= 1:
            return self.original_generator.answer(question)
        context_prompt = "Based on our conversation so far:\n"
        max_history = min(len(conversation_context) - 1, 4)
        for i in range(max(0, len(conversation_context) - max_history - 1), len(conversation_context) - 1, 2):
            user_q = conversation_context[i]["content"]
            assistant_a = conversation_context[i+1]["content"]
            context_prompt += f"You were asked: '{user_q}'\n"
            context_prompt += f"You answered: '{assistant_a}'\n"
        context_prompt += f"\nNow answer this follow-up question: {question}"
        return self.original_generator.answer(context_prompt)

# --- Page Config ---
st.set_page_config(
    page_title="Document Q&A",
    page_icon="📄",
    layout="wide"
)

# --- Session State ---
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'parsed' not in st.session_state:
    st.session_state.parsed = None
if 'selected_chunks' not in st.session_state:
    st.session_state.selected_chunks = []
if 'conversation_context' not in st.session_state:
    st.session_state.conversation_context = []

# --- Global CSS ---
st.markdown(r"""
<style>
body { background-color: #ffffff; font-family: 'Helvetica Neue', sans-serif; }
/* Chat */
.chat-container { display: flex; flex-direction: column; gap: 12px; margin: 20px 0; }
.chat-message { display: flex; }
.user-message { justify-content: flex-end; }
.assistant-message { justify-content: flex-start; }
.message-content { padding: 12px 16px; border-radius: 18px; max-width: 100%; overflow-wrap: break-word; }
.user-message .message-content { background-color: #4A90E2; color: white; border-bottom-right-radius: 4px; }
.assistant-message .message-content { background-color: #f1f1f1; color: #333; border-bottom-left-radius: 4px; }
/* Input */
.stTextInput>div>div>input { border-radius: 20px; border: 1px solid #ccc; padding: 8px 12px; }
.stButton>button { background-color: #4A90E2; color: white; border-radius: 20px; padding: 8px 16px; }
.stButton>button:hover { background-color: #357ABD; }
/* Evidence */
.evidence-content { overflow-wrap: break-word; margin-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)

# --- Sidebar Upload ---
with st.sidebar:
    st.title("Document Intelligence")
    st.image("https://img.icons8.com/ios-filled/50/4A90E2/document.png", width=40)
    st.caption(f"Last updated: {datetime.now():%Y-%m-%d}")
    st.markdown("---")
    st.subheader("Upload Document")
    uploaded_file = st.file_uploader("Select a PDF", type=["pdf"], help="Upload a PDF to analyze")
    if uploaded_file:
        filename = secure_filename(uploaded_file.name)
        if not re.match(r'^[\w\-. ]+$', filename):
            st.error("Invalid file name. Please rename your file.")
        else:
            if st.button("Parse PDF", use_container_width=True):
                output_dir = os.path.join("./parsed", filename)
                os.makedirs(output_dir, exist_ok=True)
                pdf_path = os.path.join(output_dir, filename)
                with open(pdf_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                with st.spinner("Parsing document..."):
                    try:
                        gpp = GPP(GPPConfig())
                        parsed = gpp.run(pdf_path, output_dir)
                        st.session_state.parsed = parsed
                        st.session_state.chat_history.clear()
                        st.session_state.conversation_context.clear()
                        st.session_state.selected_chunks.clear()
                        st.success("Document parsed successfully!")
                    except Exception as e:
                        st.error(f"Parsing failed: {e}")
    # removed content preview

# --- Main Area ---
main_col, evidence_col = st.columns([3, 1])
with main_col:
    st.title("Document Q&A")
    if not st.session_state.parsed:
        st.info("👈 Upload and parse a document to start")
    else:
        parsed = st.session_state.parsed
        layout_pdf = parsed.get("layout_pdf")
        if layout_pdf and os.path.exists(layout_pdf):
            st.subheader("Layout Preview")
            components.iframe(layout_pdf, height=300, width=400)
        # Chat display
        st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
        if not st.session_state.chat_history:
            st.markdown("<p style='color:#888;'>No messages yet. Start the conversation below.</p>", unsafe_allow_html=True)
        else:
            for msg in st.session_state.chat_history:
                cls = 'user-message' if msg['role']=='user' else 'assistant-message'
                st.markdown(f"<div class='chat-message {cls}'><div class='message-content'>{msg['content']}</div></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        # Input
        question = st.text_input("", key="question_input", placeholder="Type your question...", on_change=None)
        col_btn1, col_btn2 = st.columns([4, 1])
        with col_btn1:
            submit = st.button("Send", use_container_width=True)
        with col_btn2:
            clear = st.button("Clear", use_container_width=True)
        if clear:
            st.session_state.chat_history.clear()
            st.session_state.conversation_context.clear()
            st.session_state.selected_chunks.clear()
            st.experimental_rerun()
        if submit and question:
            st.session_state.chat_history.append({"role":"user","content":question})
            gen = ContextAwareAnswerGenerator(parsed['chunks'])
            answer, chunks = gen.answer(question, conversation_context=st.session_state.chat_history)
            st.session_state.chat_history.append({"role":"assistant","content":answer})
            st.session_state.selected_chunks = chunks

with evidence_col:
    if st.session_state.parsed:
        st.markdown("### Evidence")
        if not st.session_state.selected_chunks:
            st.info("Evidence appears here after asking a question.")
        else:
            for i, chunk in enumerate(st.session_state.selected_chunks,1):
                with st.expander(f"#{i}", expanded=False):
                    st.markdown(f"**Type:** {chunk.get('type','')}")
                    st.markdown(f"<div class='evidence-content'>{chunk.get('narration','')}</div>", unsafe_allow_html=True)
                    if 'table_structure' in chunk:
                        st.write(chunk['table_structure'])
                    for blk in chunk.get('blocks',[]):
                        if blk.get('type')=='img_path':
                            img_path = os.path.join(parsed['images_dir'], blk['img_path'])
                            if os.path.exists(img_path):
                                st.image(img_path, use_column_width=True)
