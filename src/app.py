import os
import streamlit as st
from datetime import datetime

from src.gpp import GPP, GPPConfig
from src.qa import AnswerGenerator

# --- Custom CSS for styling ---
st.markdown(
    """
    <style>
    body { background-color: #F5F7FA; }
    .header { text-align: center; padding: 10px; }
    .card { background: white; border-radius: 10px; padding: 15px; margin-bottom: 20px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
    .stButton>button { background-color: #4A90E2; color: white; }
    pre { background-color: #f0f0f0; padding: 10px; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True
)

# --- Page Configuration ---
st.set_page_config(
    page_title="Document Intelligence Q&A",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Header ---
st.markdown("<div class='header'>", unsafe_allow_html=True)
st.image("https://img.icons8.com/ios-filled/50/4A90E2/document.png", width=50)
st.title("Document Intelligence Q&A")
st.markdown(
    "<p style='font-size:18px; color:#555;'>Upload any PDF and get instant insights via advanced RAG-powered Q&A.</p>",
    unsafe_allow_html=True
)
st.markdown(
    f"<p style='font-size:12px; color:#888;'>Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
    unsafe_allow_html=True
)
st.markdown("</div>", unsafe_allow_html=True)

# --- Sidebar: Instructions ---
with st.sidebar:
    st.header("How It Works")
    st.markdown(
        "1. Upload and parse your PDF; 2. LLM narrates tables/images and enriches context; 3. Hybrid retrieval surfaces relevant chunks; 4. Reranker refines and generates answer."
    )
    st.markdown("---")
    st.markdown("&copy; 2025 Document Intelligence Team")

# --- Session State ---
if "parsed" not in st.session_state:
    st.session_state.parsed = None

# --- Three-Column Layout ---
col1, col2, col3 = st.columns([2, 3, 3])

# --- Left Column: Upload & Layout ---
with col1:
    st.header("1. Upload & Layout")
    uploaded_file = st.file_uploader("Select a PDF document", type=["pdf"], help="Supported: PDF files")
    if uploaded_file:
        if st.button("Parse Document"):
            output_dir = os.path.join("./parsed", uploaded_file.name)
            os.makedirs(output_dir, exist_ok=True)
            pdf_path = os.path.join(output_dir, uploaded_file.name)
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            with st.spinner("Parsing document with MinerU and LLM...⏳"):
                gpp = GPP(GPPConfig())
                parsed = gpp.run(pdf_path, output_dir)
            st.success("✅ Parsing complete!")
            st.session_state.parsed = parsed
    parsed = st.session_state.parsed
    if parsed:
        st.subheader("Layout Preview")
        layout_pdf = parsed.get("layout_pdf")
        if layout_pdf and os.path.exists(layout_pdf):
            st.markdown(f"[Open Layout PDF]({layout_pdf})")
        st.subheader("Extracted Content (Preview)")
        md_path = parsed.get("md_path")
        if md_path and os.path.exists(md_path):
            md_text = open(md_path, 'r', encoding='utf-8').read()
            st.markdown(f"<div class='card'><pre>{md_text[:2000]}{'...' if len(md_text)>2000 else ''}</pre></div>", unsafe_allow_html=True)

# --- Center Column: Q&A ---
with col2:
    st.header("2. Ask a Question")
    if parsed:
        question = st.text_input("Type your question here:", placeholder="E.g., 'What was the Q2 revenue?'" )
        if st.button("Get Answer") and question:
            with st.spinner("Retrieving answer...🤖"):
                generator = AnswerGenerator()
                answer, supporting_chunks = generator.answer(parsed['chunks'], question)
            st.markdown(f"<div class='card'><h3>Answer</h3><p>{answer}</p></div>", unsafe_allow_html=True)
            st.markdown("<div class='card'><h4>Supporting Context</h4></div>", unsafe_allow_html=True)
            for sc in supporting_chunks:
                st.write(f"- {sc['narration']}")
    else:
        st.info("Upload and parse a document to ask questions.")

# --- Right Column: Chunks ---
with col3:
    st.header("3. Relevant Chunks")
    if parsed:
        chunks = parsed.get('chunks', [])
        for idx, chunk in enumerate(chunks):
            with st.expander(f"Chunk {idx} - {chunk['type'].capitalize()}"):
                st.write(chunk.get('narration', ''))
                if 'table_structure' in chunk:
                    st.write("**Parsed Table:**")
                    st.table(chunk['table_structure'])
                for blk in chunk.get('blocks', []):
                    if blk.get('type') == 'img_path':
                        img_path = os.path.join(parsed['images_dir'], blk.get('img_path',''))
                        if os.path.exists(img_path):
                            st.image(img_path, caption=os.path.basename(img_path))
        st.info(f"Total chunks: {len(chunks)}")
    else:
        st.info("No chunks to display. Parse a document first.")
