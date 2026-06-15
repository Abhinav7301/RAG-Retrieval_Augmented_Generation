import os
import streamlit as st

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

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
    os.environ["LANGCHAIN_PROJECT"] = "testing-rag-app"

# ==========================================
# Streamlit Page Config
# ==========================================
st.set_page_config(
    page_title="Groq Q&A Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Q&A with Groq + LangChain")
st.write("Ask any question and get an AI-generated response.")

# ==========================================
# Prompt Template
# ==========================================
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful, accurate, and concise AI assistant."
        ),
        (
            "user",
            "{input}"
        )
    ]
)

# ==========================================
# Sidebar Configuration
# ==========================================
st.sidebar.header("⚙️ Model Settings")

model_name = st.sidebar.selectbox(
    "Choose Model",
    [
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile"
    ]
)

temperature = st.sidebar.slider(
    "Temperature",
    min_value=0.0,
    max_value=2.0,
    value=0.7,
    step=0.1
)

max_tokens = st.sidebar.slider(
    "Max Tokens",
    min_value=50,
    max_value=2000,
    value=500,
    step=50
)

# ==========================================
# Response Function
# ==========================================
def generate_response(question):
    llm = ChatGroq(
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        groq_api_key=GROQ_API_KEY
    )

    chain = prompt | llm

    response = chain.invoke(
        {
            "input": question
        }
    )

    return response.content

# ==========================================
# User Input
# ==========================================
user_input = st.text_area(
    "Enter your question:",
    height=120
)

# ==========================================
# Generate Button
# ==========================================
if st.button("Generate Response"):

    if not GROQ_API_KEY:
        st.error(
            "GROQ_API_KEY not found. Add it to your .env file."
        )

    elif not user_input.strip():
        st.warning(
            "Please enter a question."
        )

    else:
        try:
            with st.spinner("Generating response..."):
                answer = generate_response(user_input)

            st.subheader("Response")
            st.write(answer)

        except Exception as e:
            st.error(f"Error: {str(e)}")