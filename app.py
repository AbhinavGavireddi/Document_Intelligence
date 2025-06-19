import os
import streamlit as st
from datetime import datetime
import re
from werkzeug.utils import secure_filename
import fitz  # PyMuPDF
import base64

from src.gpp import GPP, GPPConfig
from src.qa import AnswerGenerator

# --- Page Configuration ---
st.set_page_config(
    page_title="Document Intelligence",
    page_icon="🤖",
    layout="wide"
)

# --- Session State Initialization ---
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'parsed_info' not in st.session_state:
    st.session_state.parsed_info = None  # Will store {collection_name, layout_pdf, md_path, etc.}
if "selected_chunks" not in st.session_state:
    st.session_state.selected_chunks = []

# --- Custom CSS for Messenger-like UI ---
st.markdown(
    """
    <style>
    /* Main app background */
    .stApp {
        background-color: #121212; /* Dark background */
        color: #EAEAEA; /* Light text */
    }

    /* Ensure all text in the main content area is light */
    .st-emotion-cache-16txtl3,
    .st-emotion-cache-16txtl3 h1,
    .st-emotion-cache-16txtl3 h2,
    .st-emotion-cache-16txtl3 h3 {
        color: #EAEAEA;
    }
    
    /* Sidebar adjustments */
    .st-emotion-cache-16txtl3 {
        padding-top: 2rem;
    }

    /* Main chat window container */
    .chat-window {
        height: 75vh;
        background: #1E1E1E; /* Slightly lighter dark for chat window */
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.4);
        display: flex;
        flex-direction: column;
        overflow: hidden;
    }

    /* Chat message history */
    .chat-history {
        flex-grow: 1;
        overflow-y: auto;
        padding: 20px;
        display: flex;
        flex-direction: column;
        gap: 15px;
    }

    /* General message styling */
    .message-row {
        display: flex;
        align-items: flex-end;
        gap: 10px;
    }

    /* Assistant message alignment */
    .assistant-row {
        justify-content: flex-start;
    }

    /* User message alignment */
    .user-row {
        justify-content: flex-end;
    }

    /* Avatar styling */
    .avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
        background-color: #3A3B3C; /* Dark gray for avatar */
        color: white;
    }

    /* Chat bubble styling */
    .message-bubble {
        max-width: 70%;
        padding: 10px 15px;
        border-radius: 18px;
        overflow-wrap: break-word;
        color: #EAEAEA; /* Light text for all bubbles */
    }
    
    .message-bubble p {
        margin: 0;
    }

    /* Assistant bubble color */
    .assistant-bubble {
        background-color: #3A3B3C; /* Dark gray for assistant */
    }

    /* User bubble color */
    .user-bubble {
        background-color: #0084FF;
        color: white; /* White text for user bubble */
    }

    /* Chat input container */
    .chat-input-container {
        padding: 15px 20px;
        background: #1E1E1E; /* Match chat window background */
        border-top: 1px solid #3A3B3C;
    }
    
    /* Input field styling */
    .stTextInput>div>div>input {
        border-radius: 18px;
        border: 1px solid #555;
        background-color: #3A3B3C; /* Dark input field */
        color: #EAEAEA; /* Light text in input */
        padding: 10px 15px;
    }

    /* Button styling */
    .stButton>button {
        border-radius: 18px;
        border: none;
        background-color: #0084FF;
        color: white;
        height: 42px;
    }
    
    /* Hide the default "Get Answer" header for a cleaner look */
    .st-emotion-cache-16txtl3 > h1 {
        display: none;
    }

    /* Empty chat placeholder */
    .empty-chat-placeholder {
        flex-grow: 1;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        color: #A0A0A0; /* Lighter gray for placeholder text */
    }
    
    .empty-chat-placeholder .icon {
        font-size: 50px;
        margin-bottom: 10px;
    }
    
    </style>
    """, unsafe_allow_html=True
)

# --- Left Sidebar: Instructions & Upload ---
with st.sidebar:
    # App info section
    st.image("https://img.icons8.com/ios-filled/50/4A90E2/document.png", width=40)
    st.title("Document Intelligence")
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    with st.expander("How It Works", expanded=True):
        st.markdown("1. **Upload & Parse**: Select your PDF to begin.\n2. **Ask Questions**: Use the chat to query your document.\n3. **Get Answers**: The AI provides instant, evidence-backed responses.")
    
    st.markdown("---")
    
    # Upload section
    st.subheader("Upload Document")
    uploaded_file = st.file_uploader("Select a PDF", type=["pdf"], help="Upload a PDF file to analyze")
    
    if uploaded_file:
        filename = secure_filename(uploaded_file.name)
        # Sanitize filename to be a valid Chroma collection name
        collection_name = re.sub(r'[^a-zA-Z0-9_-]', '_', os.path.splitext(filename)[0])

        if st.button("Parse Document", use_container_width=True, key="parse_button"):
            output_dir = os.path.join("./parsed", filename)
            os.makedirs(output_dir, exist_ok=True)
            pdf_path = os.path.join(output_dir, filename)
            
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            with st.spinner("Processing document..."):
                try:
                    gpp = GPP(GPPConfig())
                    parsed_info = gpp.run(pdf_path, output_dir, collection_name)
                    st.session_state.parsed_info = parsed_info
                    st.session_state.chat_history = []
                    st.session_state.selected_chunks = []
                    st.success("Document ready!")
                except Exception as e:
                    st.error(f"Processing failed: {str(e)}")
                    st.session_state.parsed_info = None

    # Display document preview if parsed
    if st.session_state.parsed_info:
        st.markdown("---")
        st.subheader("Document Preview")
        parsed = st.session_state.parsed_info
        
        # Layout PDF
        layout_pdf = parsed.get("layout_pdf")
        if layout_pdf and os.path.exists(layout_pdf):
            with st.expander("View Layout PDF", expanded=False):
                st.markdown(f"[Open in new tab]({layout_pdf})")
                doc = fitz.open(layout_pdf)
                thumb_width = 500
                thumbs = []
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    pix = page.get_pixmap(matrix=fitz.Matrix(thumb_width / page.rect.width, thumb_width / page.rect.width))
                    img_bytes = pix.tobytes("png")
                    b64 = base64.b64encode(img_bytes).decode("utf-8")
                    thumbs.append((page_num, b64))
                st.markdown("<div style='overflow-x: auto; white-space: nowrap; border: 1px solid #eee; border-radius: 8px; padding: 8px; background: #fafbfc; max-width: 100%;'>", unsafe_allow_html=True)
                for page_num, b64 in thumbs:
                    st.markdown(f"<a href='{layout_pdf}#page={page_num+1}' target='_blank' style='display:inline-block;margin-right:8px;'><img src='data:image/png;base64,{b64}' width='{thumb_width}' style='border:1px solid #ccc;border-radius:4px;box-shadow:0 1px 2px #0001;'/></a>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
        
        # Content preview
        md_path = parsed.get("md_path")
        if md_path and os.path.exists(md_path):
            try:
                with open(md_path, 'r', encoding='utf-8') as md_file:
                    md_text = md_file.read()
                with st.expander("Content Preview", expanded=False):
                    st.markdown(f"<pre style='font-size:12px;max-height:300px;overflow-y:auto'>{md_text[:3000]}{'...' if len(md_text)>3000 else ''}</pre>", unsafe_allow_html=True)
            except Exception as e:
                st.warning(f"Could not preview content: {str(e)}")

    st.markdown("---")
    st.subheader("Chat Controls")
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.selected_chunks = []
        st.rerun()

# --- Main Chat Area ---
main_col, evidence_col = st.columns([2, 1])

with main_col:
    if not st.session_state.parsed_info:
        st.info("Please upload and parse a document to start the chat.")
    else:
        # Create a container for the chat window
        st.markdown("<div class='chat-window'>", unsafe_allow_html=True)
        
        # Display chat history
        st.markdown("<div class='chat-history'>", unsafe_allow_html=True)
        if not st.session_state.chat_history:
             st.markdown("""
            <div class='empty-chat-placeholder'>
                <span class="icon">🤖</span>
                <h3>Ask me anything about your document!</h3>
            </div>
            """, unsafe_allow_html=True)
        else:
            for message in st.session_state.chat_history:
                if message["role"] == "user":
                    st.markdown(f"""
                    <div class="message-row user-row">
                        <div class="message-bubble user-bubble">
                            <p>{message["content"]}</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="message-row assistant-row">
                        <div class="avatar">🤖</div>
                        <div class="message-bubble assistant-bubble">
                            <p>{message["content"]}</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True) # Close chat-history
        
        # Chat input bar
        st.markdown("<div class='chat-input-container'>", unsafe_allow_html=True)
        input_col, button_col = st.columns([4, 1])
        with input_col:
            question = st.text_input("Ask a question...", key="question_input", label_visibility="collapsed")
        with button_col:
            send_button = st.button("Send", use_container_width=True)
            
        st.markdown("</div>", unsafe_allow_html=True) # Close chat-input-container
        st.markdown("</div>", unsafe_allow_html=True) # Close chat-window

        # --- Handle message sending ---
        if send_button and question:
            st.session_state.chat_history.append({"role": "user", "content": question})
            
            with st.spinner("Thinking..."):
                generator = AnswerGenerator(st.session_state.parsed_info['collection_name'])
                answer, supporting_chunks = generator.answer(question)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                st.session_state.selected_chunks = supporting_chunks
            
            st.rerun()

# --- Supporting Evidence in the right column ---
with evidence_col:
    if st.session_state.parsed_info:
        st.markdown("### Supporting Evidence")
        
        if not st.session_state.selected_chunks:
            st.info("Evidence chunks will appear here after you ask a question.")
        else:
            for idx, chunk in enumerate(st.session_state.selected_chunks):
                with st.expander(f"Evidence Chunk #{idx+1}", expanded=True):
                    st.markdown(chunk.get('narration', 'No narration available'))
                    if 'table_structure' in chunk:
                        st.dataframe(chunk['table_structure'], use_container_width=True)
                    for blk in chunk.get('blocks', []):
                        if blk.get('type') == 'img_path' and 'images_dir' in st.session_state.parsed_info:
                            img_path = os.path.join(st.session_state.parsed_info['images_dir'], blk.get('img_path',''))
                            if os.path.exists(img_path):
                                st.image(img_path, use_column_width=True)

# -- Error handling wrapper -- 
def handle_error(func):
    try:
        func()
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        st.info("Please refresh the page and try again.")

# Wrap the entire app in the error handler
handle_error(lambda: None)