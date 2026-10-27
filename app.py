import os
import streamlit as st
from openai import OpenAI

# Page configuration
st.set_page_config(
    page_title="Professional AI Assistant",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Initialize chat history before using sidebar buttons
if "messages" not in st.session_state:
    st.session_state.messages = []

# Custom UI styling
st.markdown("""
<style>
    .stChatInputContainer {
        padding-bottom: 20px;
    }
    div[data-testid="stChatMessage"] {
        padding: 1rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image(
        "https://img.icons8.com/color/96/artificial-intelligence.png",
        width=80
    )
    st.title("Control Panel")
    st.markdown("---")
    st.markdown("### About")
    st.info(
        "This is an AI-powered assistant built with "
        "Streamlit and Numan Ali"
    )

    if st.button("Clear Conversation History"):
        st.session_state.messages = []
        st.rerun()

# Main header
st.title("🤖 Enterprise AI Assistant")
st.caption("Your intelligent workspace companion powered by Numan Ali.")
st.markdown("---")

# Azure OpenAI configuration
API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

if not API_KEY or not ENDPOINT or not DEPLOYMENT:
    st.error(
        "Azure environment settings are missing. "
        "Please check your Azure App Settings."
    )
    st.stop()

client = OpenAI(
    api_key=API_KEY,
    base_url=ENDPOINT.rstrip("/") + "/"
)

# Function to check if query is related to Pakistan history
def is_related_to_pakistan_history(text):
    keywords = [
        "pakistan", "history", "muhammad ali jinnah", "pakistan movement",
        "lahore resolution", "independence", "1947", "partition", "quaid-e-azam",
        "allama iqbal", "muslim league", "kashmir", "pak army"
    ]
    text_lower = text.lower()
    for kw in keywords:
        if kw in text_lower:
            return True
    return False

# Display conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Type your question here..."):
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Check if prompt is related to Pakistan history
                if is_related_to_pakistan_history(prompt):
                    # AI se jawab lein
                    response = client.chat.completions.create(
                        model=DEPLOYMENT,
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You are an assistant created by Numan Ali. "
                                    "When asked who created you, always say: "
                                    "'I was created by Numan Ali, an artificial intelligence research organization. "
                                    "How can I assist you further?'"
                                )
                            },
                            *st.session_state.messages
                        ],
                    )
                    answer = response.choices[0].message.content

                    if not answer:
                        answer = "Received an empty response from the AI."

                else:
                    # Domain se bahar sawal hai
                    answer = "Sorry, this is outside my domain."

                st.markdown(answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )

            except Exception as e:
                st.error(f"Azure OpenAI Error: {e}")
                # Remove last user message to keep chat state consistent
                st.session_state.messages.pop()