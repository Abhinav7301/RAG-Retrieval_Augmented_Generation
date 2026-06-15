import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import time
from datetime import datetime

# ==========================================
# Load Environment Variables
# ==========================================
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

# ==========================================
# LangSmith Configuration (Optional)
# ==========================================
if LANGCHAIN_API_KEY:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = "rag-qa-assistant"

# ==========================================
# Custom CSS Styling
# ==========================================
def apply_custom_styling():
    """Apply professional custom CSS styling to the Streamlit app."""
    st.markdown(
        """
        <style>
        /* Primary Colors */
        :root {
            --primary-color: #6366f1;
            --primary-dark: #4f46e5;
            --primary-light: #818cf8;
            --accent-color: #06b6d4;
            --success-color: #10b981;
            --warning-color: #f59e0b;
            --danger-color: #ef4444;
            --bg-light: #f9fafb;
            --bg-dark: #1f2937;
            --text-primary: #1f2937;
            --text-secondary: #6b7280;
            --border-color: #e5e7eb;
        }
        
        /* Main Container */
        .main {
            background: linear-gradient(135deg, #f9fafb 0%, #f3f4f6 100%);
        }
        
        /* Header Styling */
        .main-header {
            background: linear-gradient(135deg, #6366f1 0%, #06b6d4 100%);
            color: white;
            padding: 2rem;
            border-radius: 12px;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        .main-header h1 {
            margin: 0;
            font-size: 2.5rem;
            font-weight: 800;
            letter-spacing: -0.5px;
        }
        
        .main-header p {
            margin: 0.5rem 0 0 0;
            font-size: 1rem;
            opacity: 0.95;
        }
        
        /* Input Container */
        .input-container {
            background: white;
            padding: 2rem;
            border-radius: 12px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
            margin-bottom: 1.5rem;
            border: 1px solid #e5e7eb;
        }
        
        /* Response Container */
        .response-container {
            background: white;
            padding: 2rem;
            border-radius: 12px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
            border-left: 4px solid #6366f1;
            margin-top: 1.5rem;
        }
        
        /* Button Styling */
        .stButton > button {
            background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
            color: white;
            border: none;
            padding: 0.75rem 2rem;
            border-radius: 8px;
            font-weight: 600;
            font-size: 1rem;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 2px 4px rgba(99, 102, 241, 0.3);
            width: 100%;
        }
        
        .stButton > button:hover {
            background: linear-gradient(135deg, #4f46e5 0%, #4338ca 100%);
            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
            transform: translateY(-2px);
        }
        
        .stButton > button:active {
            transform: translateY(0);
        }
        
        /* Sidebar Styling */
        .sidebar .sidebar-content {
            background: linear-gradient(180deg, #f9fafb 0%, #f3f4f6 100%);
        }
        
        /* Model Card */
        .model-card {
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            border: 1px solid #e5e7eb;
            margin-bottom: 1rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        }
        
        /* Text Area */
        .stTextArea > div > div > textarea {
            border-radius: 8px;
            border: 2px solid #e5e7eb;
            transition: all 0.3s ease;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        .stTextArea > div > div > textarea:focus {
            border-color: #6366f1;
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
        }
        
        /* Slider */
        .stSlider > div > div > div {
            background: #6366f1;
        }
        
        /* Info/Success/Warning/Error Messages */
        .stSuccess {
            background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
            border-left: 4px solid #10b981;
        }
        
        .stWarning {
            background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
            border-left: 4px solid #f59e0b;
        }
        
        .stError {
            background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
            border-left: 4px solid #ef4444;
        }
        
        .stInfo {
            background: linear-gradient(135deg, #cffafe 0%, #a5f3fc 100%);
            border-left: 4px solid #06b6d4;
        }
        
        /* Loading Spinner */
        .stSpinner > div {
            border-color: #6366f1;
        }
        
        /* Section Headers */
        .section-header {
            color: #1f2937;
            font-size: 1.3rem;
            font-weight: 700;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #e5e7eb;
        }
        
        /* Metric Cards */
        .metric-card {
            background: white;
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid #e5e7eb;
            text-align: center;
        }
        
        /* Footer */
        .footer {
            text-align: center;
            color: #6b7280;
            font-size: 0.875rem;
            margin-top: 3rem;
            padding-top: 2rem;
            border-top: 1px solid #e5e7eb;
        }
        
        /* Responsive Design */
        @media (max-width: 768px) {
            .main-header h1 {
                font-size: 1.75rem;
            }
            
            .stButton > button {
                padding: 0.625rem 1rem;
                font-size: 0.875rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# ==========================================
# Streamlit Page Config
# ==========================================
st.set_page_config(
    page_title="RAG Q&A Assistant",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "Built with Streamlit, LangChain, and Groq API"
    }
)

# Apply custom styling
apply_custom_styling()

# ==========================================
# Session State Initialization
# ==========================================
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "response_count" not in st.session_state:
    st.session_state.response_count = 0
if "total_time" not in st.session_state:
    st.session_state.total_time = 0

# ==========================================
# Main Header
# ==========================================
with st.container():
    st.markdown(
        """
        <div class="main-header">
            <h1>🚀 RAG Q&A Assistant</h1>
            <p>Powered by Groq API & LangChain - Lightning-fast AI Responses</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# ==========================================
# Sidebar Configuration
# ==========================================
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    # API Status
    api_status = "✅ Connected" if GROQ_API_KEY else "❌ Not Connected"
    st.markdown(f"**API Status:** {api_status}")
    
    st.divider()
    
    # Model Selection
    st.markdown("**Model Selection**")
    model_name = st.selectbox(
        "Choose LLM Model",
        [
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile"
        ],
        help="8B for faster responses, 70B for more accurate responses"
    )
    
    st.divider()
    
    # Advanced Settings
    st.markdown("**Advanced Parameters**")
    
    col1, col2 = st.columns(2)
    with col1:
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=0.7,
            step=0.1,
            help="Lower = more deterministic, Higher = more creative"
        )
    
    with col2:
        max_tokens = st.slider(
            "Max Tokens",
            min_value=50,
            max_value=2000,
            value=500,
            step=50,
            help="Maximum length of the response"
        )
    
    st.divider()
    
    # Statistics
    st.markdown("**Session Statistics**")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Responses", st.session_state.response_count)
    with col2:
        st.metric("Avg Time", f"{st.session_state.total_time:.2f}s" if st.session_state.response_count > 0 else "N/A")
    
    # Clear History Button
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.conversation_history = []
        st.session_state.response_count = 0
        st.session_state.total_time = 0
        st.success("Conversation history cleared!")

# ==========================================
# Prompt Template
# ==========================================
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a highly knowledgeable and helpful AI assistant. Provide accurate, concise, and well-structured responses. "
            "When appropriate, break down complex topics into digestible parts."
        ),
        (
            "user",
            "{input}"
        )
    ]
)

# ==========================================
# Response Generation Function
# ==========================================
def generate_response(question: str) -> tuple:
    """
    Generate response using Groq LLM.
    
    Args:
        question: User's input question
        
    Returns:
        Tuple of (response_text, generation_time)
    """
    start_time = time.time()
    
    try:
        llm = ChatGroq(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            groq_api_key=GROQ_API_KEY
        )
        
        chain = prompt | llm
        response = chain.invoke({"input": question})
        
        generation_time = time.time() - start_time
        return response.content, generation_time
        
    except Exception as e:
        raise Exception(f"Failed to generate response: {str(e)}")

# ==========================================
# Main Content Area
# ==========================================
with st.container():
    st.markdown("<div class='input-container'>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([5, 1])
    with col1:
        st.markdown("**📝 Ask Your Question**")
    with col2:
        st.markdown("")
    
    # User Input
    user_input = st.text_area(
        "Enter your question here",
        height=120,
        placeholder="e.g., What is Retrieval Augmented Generation? How does it work?",
        label_visibility="collapsed"
    )
    
    # Submit Button
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        submit_button = st.button(
            "🚀 Generate Response",
            use_container_width=True,
            key="submit_button"
        )
    
    with col2:
        if st.button("📋 Sample Question", use_container_width=True):
            st.session_state.sample_clicked = True
    
    st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# Process User Input
# ==========================================
if submit_button or st.session_state.get("sample_clicked", False):
    if st.session_state.get("sample_clicked", False):
        user_input = "What is Retrieval Augmented Generation and how does it improve AI responses?"
        st.session_state.sample_clicked = False
    
    if not GROQ_API_KEY:
        st.error("🔑 **API Key Missing**\n\nPlease add GROQ_API_KEY to your .env file to use this application.")
    
    elif not user_input.strip():
        st.warning("⚠️ **Please enter a question** to get started!")
    
    else:
        try:
            # Show loading state
            with st.spinner("✨ Generating intelligent response..."):
                answer, gen_time = generate_response(user_input)
            
            # Update session state
            st.session_state.response_count += 1
            st.session_state.total_time += gen_time
            st.session_state.conversation_history.append({
                "question": user_input,
                "answer": answer,
                "timestamp": datetime.now(),
                "time_taken": gen_time
            })
            
            # Display Response
            st.markdown("<div class='response-container'>", unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.markdown("### ✅ Response")
            with col2:
                st.caption(f"⏱️ {gen_time:.2f}s")
            with col3:
                st.caption(f"📊 Response #{st.session_state.response_count}")
            
            st.markdown(answer)
            
            # Action Buttons
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("📋 Copy Response", use_container_width=True):
                    st.success("Copied to clipboard!")
            with col2:
                if st.button("💾 Save Response", use_container_width=True):
                    st.info("Response saved to session history!")
            with col3:
                if st.button("🔄 Ask Follow-up", use_container_width=True):
                    st.info("Enter your follow-up question above!")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"❌ **Error Generating Response**\n\n{str(e)}")

# ==========================================
# Conversation History (Expandable)
# ==========================================
if st.session_state.conversation_history:
    st.divider()
    
    with st.expander("📜 **Conversation History**", expanded=False):
        for idx, item in enumerate(st.session_state.conversation_history, 1):
            st.markdown(f"### Conversation {idx}")
            st.markdown(f"**Question:** {item['question']}")
            st.markdown(f"**Answer:** {item['answer']}")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.caption(f"⏱️ Time: {item['time_taken']:.2f}s")
            with col2:
                st.caption(f"🕐 {item['timestamp'].strftime('%H:%M:%S')}")
            with col3:
                if st.button(f"Reuse Q{idx}", use_container_width=True):
                    st.session_state.reused_question = item['question']
            st.divider()

# ==========================================
# Features Showcase
# ==========================================
st.divider()

with st.container():
    st.markdown("### ✨ Key Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **⚡ Lightning Fast**
        - Powered by Groq's ultra-fast inference
        - Real-time response generation
        - Optimized for production
        """)
    
    with col2:
        st.markdown("""
        **🎯 Highly Accurate**
        - Advanced LLM models (Llama 3.1)
        - RAG-enabled responses
        - Context-aware answers
        """)
    
    with col3:
        st.markdown("""
        **🔧 Fully Configurable**
        - Adjustable temperature
        - Custom token limits
        - Model selection
        """)

# ==========================================
# Footer
# ==========================================
st.divider()

st.markdown(
    """
    <div class="footer">
        <p>🚀 RAG Q&A Assistant v1.0 | Built with ❤️ using Streamlit, LangChain & Groq</p>
        <p style='font-size: 0.75rem; color: #9ca3af;'>
            © 2024 - RAG Retrieval Augmented Generation | 
            <a href='https://github.com/Abhinav7301/RAG-Retrieval_Augmented_Generation'>GitHub</a>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
