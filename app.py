import os
import streamlit as st
from datetime import datetime
import re
from werkzeug.utils import secure_filename

from src.gpp import GPP, GPPConfig
from src.qa import AnswerGenerator

# Check if we need to modify the AnswerGenerator class to accept conversation context
# If the original implementation doesn't support this, we'll create a wrapper

class ContextAwareAnswerGenerator:
    """Wrapper around AnswerGenerator to include conversation context"""
    
    def __init__(self, chunks):
        self.chunks = chunks
        self.original_generator = AnswerGenerator(chunks)
    
    def answer(self, question, conversation_context=None):
        """
        Generate answer with conversation context
        
        Args:
            chunks: Document chunks to search
            question: Current question
            conversation_context: List of previous Q&A for context
            
        Returns:
            answer, supporting_chunks
        """
        # If no conversation context or original implementation supports it directly
        if conversation_context is None or len(conversation_context) <= 1:
            return self.original_generator.answer(question)
            
        # Otherwise, enhance the question with context
        # Create a contextual prompt by summarizing previous exchanges
        context_prompt = "Based on our conversation so far:\n"
        
        # Include the last few exchanges (limiting to prevent context getting too large)
        max_history = min(len(conversation_context) - 1, 4)  # Last 4 exchanges maximum
        for i in range(max(0, len(conversation_context) - max_history - 1), len(conversation_context) - 1, 2):
            if i < len(conversation_context) and i+1 < len(conversation_context):
                user_q = conversation_context[i]["content"]
                assistant_a = conversation_context[i+1]["content"]
                context_prompt += f"You were asked: '{user_q}'\n"
                context_prompt += f"You answered: '{assistant_a}'\n"
        
        context_prompt += f"\nNow answer this follow-up question: {question}"
        
        # Use the enhanced prompt
        return self.original_generator.answer(context_prompt)

# --- Page Configuration ---
st.set_page_config(
    page_title="Document Intelligence Q&A",
    page_icon="📄",
    layout="wide"
)

# --- Session State Initialization ---
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []  # List of {role: 'user'/'assistant', content: str}
if 'parsed' not in st.session_state:
    st.session_state.parsed = None
if "selected_chunks" not in st.session_state:
    st.session_state.selected_chunks = []
if "conversation_context" not in st.session_state:
    st.session_state.conversation_context = []

# --- Custom CSS for styling ---
st.markdown(
    """
    <style>
    /* Global Styles */
    body {
        background-color: #fafafa;
        font-family: 'Helvetica Neue', sans-serif;
    }
    
    /* Header Styles */
    .main-header {
        margin-bottom: 2rem;
    }
    
    /* Card Styles */
    .card {
        background: white;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
    }
    
    /* Button Styles */
    .stButton>button {
        background-color: #4361ee;
        color: white;
        border-radius: 4px;
        border: none;
        padding: 8px 16px;
        font-weight: 500;
    }
    
    .stButton>button:hover {
        background-color: #3a56d4;
    }
    
    /* Input Styles */
    .stTextInput>div>div>input {
        border-radius: 4px;
        border: 1px solid #e0e0e0;
    }
    
    /* Code Block Styles */
    pre {
        background-color: #f5f5f5;
        padding: 12px;
        border-radius: 4px;
        font-size: 14px;
    }
    
    /* Hide Streamlit footer */
    footer {
        display: none;
    }
    
    /* Sidebar Styles */
    .css-18e3th9 {
        padding-top: 1rem;
    }
    
    /* Expander styles */
    .streamlit-expanderHeader {
        font-size: 1rem;
        font-weight: 500;
    }
    
    /* Chat Interface Styles */
    .chat-container {
        display: flex;
        flex-direction: column;
        gap: 12px;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    
    .chat-message {
        display: flex;
        margin-bottom: 10px;
    }
    
    .user-message {
        justify-content: flex-end;
    }
    
    .assistant-message {
        justify-content: flex-start;
    }
    
    .message-content {
        padding: 12px 16px;
        border-radius: 18px;
        max-width: 80%;
        overflow-wrap: break-word;
    }
    
    .user-message .message-content {
        background-color: #4361ee;
        color: white;
        border-bottom-right-radius: 4px;
    }
    
    .assistant-message .message-content {
        background-color: #f0f2f6;
        color: #1e1e1e;
        border-bottom-left-radius: 4px;
    }
    
    .message-content p {
        margin: 0;
        padding: 0;
    }
    
    /* Empty chat placeholder style */
    .empty-chat-placeholder {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 300px;
        background-color: #f8f9fa;
        border-radius: 8px;
        margin-bottom: 20px;
        text-align: center;
        color: #6c757d;
    }
    
    .empty-chat-icon {
        font-size: 40px;
        margin-bottom: 16px;
        color: #adb5bd;
    }
    
    /* Message typing indicator */
    .typing-indicator {
        display: flex;
        align-items: center;
        justify-content: flex-start;
        margin-top: 8px;
    }
    
    .typing-indicator span {
        height: 8px;
        width: 8px;
        background-color: #4361ee;
        border-radius: 50%;
        margin: 0 2px;
        display: inline-block;
        opacity: 0.7;
    }
    
    .typing-indicator span:nth-child(1) {
        animation: pulse 1s infinite;
    }
    
    .typing-indicator span:nth-child(2) {
        animation: pulse 1s infinite 0.2s;
    }
    
    .typing-indicator span:nth-child(3) {
        animation: pulse 1s infinite 0.4s;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); opacity: 0.7; }
        50% { transform: scale(1.2); opacity: 1; }
        100% { transform: scale(1); opacity: 0.7; }
    }
    
    /* Spinner */
    .stSpinner > div > div {
        border-top-color: #4361ee !important;
    }
    
    /* Info box */
    .stAlert {
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True
)

# --- Left Sidebar: Instructions & Upload ---
with st.sidebar:
    # App info section
    st.image("https://img.icons8.com/ios-filled/50/4A90E2/document.png", width=40)
    st.title("Document Intelligence")
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d')}")
    
    with st.expander("How It Works", expanded=True):
        st.markdown(
            """
            1. **Upload PDF**: Select and parse your document
            2. **Ask Questions**: Type your query about the document
            3. **Get Answers**: AI analyzes and responds with insights
            4. **View Evidence**: See supporting chunks in the right sidebar
            """
        )
    
    st.markdown("---")
    
    # Upload section
    st.subheader("Upload Document")
    uploaded_file = st.file_uploader("Select a PDF", type=["pdf"], help="Upload a PDF file to analyze")
    
    if uploaded_file:
        try:
            filename = secure_filename(uploaded_file.name)
            if not re.match(r'^[\w\-. ]+$', filename):
                st.error("Invalid file name. Please rename your file.")
            else:
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Parse pdf", use_container_width=True, key="parse_button"):
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
                                st.session_state.chat_history = []  # Reset chat when new document is parsed
                                st.session_state.conversation_context = []  # Reset conversation context
                                st.session_state.selected_chunks = []  # Reset selected chunks
                                st.success("Document parsed successfully!")
                            except Exception as e:
                                st.error(f"Parsing failed: {str(e)}")
                                st.session_state.parsed = None
                with col2:
                    if st.button("Clear", use_container_width=True, key="clear_button"):
                        st.session_state.parsed = None
                        st.session_state.selected_chunks = []
                        st.session_state.chat_history = []
                        st.session_state.conversation_context = []
                        st.experimental_rerun()
        except Exception as e:
            st.error(f"Upload error: {str(e)}")
    
    # Display document preview if parsed
    if st.session_state.parsed:
        st.markdown("---")
        st.subheader("Document Preview")
        parsed = st.session_state.parsed
        
        # Layout PDF
        layout_pdf = parsed.get("layout_pdf")
        if layout_pdf and os.path.exists(layout_pdf):
            with st.expander("View Layout PDF", expanded=False):
                st.markdown(f"[Open in new tab]({layout_pdf})")
        
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

# --- Main Content Area ---
# Create a two-column layout for main content
main_col, evidence_col = st.columns([3, 1])

with main_col:
    st.markdown("<div class='main-header'>", unsafe_allow_html=True)
    st.title("Document Q&A")
    st.markdown("</div>", unsafe_allow_html=True)

    if not st.session_state.parsed:
        st.info("👈 Please upload and parse a document to begin asking questions.")
    else:
        # Q&A Section with chat-like interface
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        question =     st.text_input(
            "Ask a question about your document:",
            key="question_input",
            placeholder="E.g., 'What are the key findings?' or 'Summarize the data'",
            on_change=None  # Ensure the input field gets cleared naturally after submission
        )
    
    col_btn1, col_btn2 = st.columns([4, 1])
    with col_btn1:
        submit_button = st.button("Get Answer", use_container_width=True)
    with col_btn2:
        clear_chat = st.button("Clear Chat", use_container_width=True)
    
    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        
    # Clear chat when button is pressed
    if clear_chat:
        st.session_state.chat_history = []
        st.session_state.conversation_context = []
        st.session_state.selected_chunks = []
        st.experimental_rerun()
        
    if submit_button and question:
        with st.spinner("Analyzing document and generating answer..."):
            try:
                # Add user question to chat history
                st.session_state.chat_history.append({"role": "user", "content": question})
                
                # Generate answer using conversation context
                generator = ContextAwareAnswerGenerator(st.session_state.parsed['chunks'])
                answer, supporting_chunks = generator.answer(
                    question, conversation_context=st.session_state.chat_history
                )
                
                # Add assistant response to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                
                # Store supporting chunks in session state for the right sidebar
                st.session_state.selected_chunks = supporting_chunks
                
                # Clear the question input
                question = ""
                
            except Exception as e:
                st.error(f"Failed to generate answer: {str(e)}")
                st.session_state.selected_chunks = []
                
    # Display chat history
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    
    if not st.session_state.chat_history:
        # Show empty chat state with icon
        st.markdown("""
        <div class='empty-chat-placeholder'>
            <div class='empty-chat-icon'>💬</div>
            <p>Ask questions about your document to start a conversation</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"""
                <div class='chat-message user-message'>
                    <div class='message-content'>
                        <p>{message["content"]}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='chat-message assistant-message'>
                    <div class='message-content'>
                        <p>{message["content"]}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# --- Supporting Evidence in the right column ---
with evidence_col:
    if st.session_state.parsed:
        st.markdown("### Supporting Evidence")
        
        if not st.session_state.selected_chunks:
            st.info("Evidence chunks will appear here after you ask a question.")
        else:
            for idx, chunk in enumerate(st.session_state.selected_chunks):
                with st.expander(f"Evidence #{idx+1}", expanded=True):
                    st.markdown(f"**Type:** {chunk['type'].capitalize()}")
                    st.markdown(chunk.get('narration', 'No narration available'))
                    
                    # Display table if available
                    if 'table_structure' in chunk:
                        st.write("**Table Data:**")
                        st.dataframe(chunk['table_structure'], use_container_width=True)
                    
                    # Display images if available
                    for blk in chunk.get('blocks', []):
                        if blk.get('type') == 'img_path' and 'images_dir' in st.session_state.parsed:
                            img_path = os.path.join(st.session_state.parsed['images_dir'], blk.get('img_path',''))
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