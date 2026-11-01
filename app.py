import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, AzureOpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
load_dotenv()

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Pakistan History AI",
    page_icon="🇵🇰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# 2. SESSION STATE
# --------------------------------------------------
if "messages" not in st.session_state:
  st.session_state.messages = []

if "chat_started" not in st.session_state:
  st.session_state.chat_started = False

# --------------------------------------------------
# 3. CUSTOM CSS - CHATGPT-INSPIRED UI
# --------------------------------------------------
st.markdown(
    """
<style>
    #MainMenu, footer, header {
        visibility: hidden;
    }

    [data-testid="stAppViewContainer"] {
        background: var(--background-color);
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(128,128,128,0.2);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.5rem;
    }

    .brand-title {
        font-size: 1.35rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        padding-bottom: 0.2rem;
    }

    .brand-subtitle {
        font-size: 0.85rem;
        opacity: 0.7;
        margin-bottom: 1rem;
    }

    .main-heading {
        font-size: 2rem;
        font-weight: 650;
        letter-spacing: -0.8px;
        text-align: center;
        margin-top: 1.8rem;
        margin-bottom: 0.3rem;
    }

    .main-caption {
        text-align: center;
        opacity: 0.7;
        margin-bottom: 2rem;
    }

    [data-testid="stChatMessage"] {
        border-radius: 16px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.8rem;
    }

    [data-testid="stChatInput"] {
        border-radius: 24px;
    }

    [data-testid="stChatInput"] textarea {
        padding-top: 0.8rem;
    }

    .footer-note {
        text-align: center;
        opacity: 0.55;
        font-size: 0.75rem;
        margin-top: 0.6rem;
        padding-bottom: 1rem;
    }

    div.stButton > button {
        border-radius: 12px;
        min-height: 42px;
    }

    @media (max-width: 768px) {
        .main-heading {
            font-size: 1.6rem;
        }
    }
</style>
""",
    unsafe_allow_html=True,
)

# --------------------------------------------------
# 4. AZURE OPENAI CONFIGURATION
# --------------------------------------------------
API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
DEPLOYMENT = os.getenv(
    "AZURE_OPENAI_DEPLOYMENT_NAME"
)
EMBEDDING_DEPLOYMENT = os.getenv(
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small"
)

if not API_KEY or not ENDPOINT or not DEPLOYMENT:
  st.error(
      "Azure settings missing. Please configure API_KEY, ENDPOINT and"
      " DEPLOYMENT_NAME."
  )
  st.stop()


# --------------------------------------------------
# 5. INITIALIZE RAG (VECTOR STORE & RETRIEVER)
# --------------------------------------------------
@st.cache_resource
def load_rag_retriever():
  pdf_path = "Pakistan_History_Knowledge_Base.pdf"
  if not os.path.exists(pdf_path):
    st.error(
        f"Error: '{pdf_path}' file nahi mili! Baraye meharbani PDF file ko project"
        " directory mein rakhein."
    )
    return None

  try:
    # PDF Load karein
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Text ko Chunks mein split karein
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    )
    docs = text_splitter.split_documents(documents)

    # Embeddings initialize karein (Azure OpenAI)
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=EMBEDDING_DEPLOYMENT,
        openai_api_version="2023-05-15",
        azure_endpoint=ENDPOINT,
        api_key=API_KEY,
    )

    # Vector store banayein (FAISS)
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore.as_retriever(
        search_kwargs={"k": 3}
    ) 
# Top 3 relevant chunks fetch karega
  except Exception as e:
    st.error(f"RAG Initialization Error: {e}")
    return None


retriever = load_rag_retriever()

# --------------------------------------------------
# 6. CHATBOT PERSONALITY AND DOMAIN RULES
# --------------------------------------------------
SYSTEM_PROMPT = """
You are Pakistan History AI, an assistant created by Numan Ali.

YOUR IDENTITY:
- If asked who created, developed, built, or made you,
  say that you were created by Numan Ali.
- You may explain that you are an AI assistant designed
  to help people learn about Pakistan's history using verified knowledge bases.
- Never claim that Numan Ali is an organization.
- Do not invent personal details about your creator.

INSTRUCTIONS & RAG CONTEXT:
- You will be provided with retrieved context snippets from an authoritative Pakistan history document.
- Use the provided context to answer questions accurately and concisely.
- Understand English, Urdu and Roman Urdu.
- Reply in the language used by the user when practical (if Roman Urdu, reply in Roman Urdu).
- If the answer is not present in the context or general Pakistan history domain, politely explain that you specialize in Pakistan's history.

OUT-OF-DOMAIN RULE:
If the user asks for help with unrelated topics such as programming, mathematics, general science, sports, shopping, recipes, or unrelated current affairs, do not answer the actual question. Politely invite them to ask about Pakistan's history.
"""

# --------------------------------------------------
# 7. SIDEBAR
# --------------------------------------------------
with st.sidebar:
  st.markdown(
      '<div class="brand-title">🇵🇰 Pakistan History AI</div>',
      unsafe_allow_html=True,
  )
  st.markdown(
      '<div class="brand-subtitle">Your historical knowledge companion</div>',
      unsafe_allow_html=True,
  )

  if st.button("＋  New chat", use_container_width=True):
    st.session_state.messages = []
    st.session_state.chat_started = False
    st.rerun()

  st.markdown("---")
  st.markdown("### About this assistant")
  st.caption(
      "Explore Pakistan's history, its founding leaders, the Pakistan Movement"
      " and important historical events using RAG search."
  )

  st.markdown("---")
  st.markdown("### Your assistant")
  st.markdown("**Created by:** Numan Ali")
  st.markdown("**Platform:** Azure OpenAI + RAG")

  st.markdown("---")

  if st.button("🗑️ Clear conversation", use_container_width=True):
    st.session_state.messages = []
    st.session_state.chat_started = False
    st.rerun()

# --------------------------------------------------
# 8. MAIN HEADER
# --------------------------------------------------
st.markdown(
    '<div class="main-heading">What would you like to know?</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="main-caption">Ask anything about Pakistan history, or say'
    " hello.</div>",
    unsafe_allow_html=True,
)

# --------------------------------------------------
# 9. SUGGESTED PROMPTS
# --------------------------------------------------
if not st.session_state.messages:
  st.markdown("")

  col1, col2 = st.columns(2)

  with col1:
    if st.button("🇵🇰  Who founded Pakistan?", use_container_width=True):
      st.session_state.pending_prompt = "Who founded Pakistan?"
      st.rerun()

    if st.button("📜  Explain the Lahore Resolution", use_container_width=True):
      st.session_state.pending_prompt = "Explain the Lahore Resolution of 1940."
      st.rerun()

  with col2:
    if st.button("👤  Who created you?", use_container_width=True):
      st.session_state.pending_prompt = "Who created you?"
      st.rerun()

    if st.button("🗓️  Why was Pakistan created?", use_container_width=True):
      st.session_state.pending_prompt = "Why was Pakistan created in 1947?"
      st.rerun()

# --------------------------------------------------
# 10. DISPLAY CHAT HISTORY
# --------------------------------------------------
for message in st.session_state.messages:
  with st.chat_message(message["role"]):
    st.markdown(message["content"])

# --------------------------------------------------
# 11. HANDLE USER INPUT & RAG EXECUTION
# --------------------------------------------------
prompt = st.chat_input("Message Pakistan History AI...")

# Handle suggested prompt buttons
if "pending_prompt" in st.session_state:
  prompt = st.session_state.pop("pending_prompt")

if prompt:
  st.session_state.chat_started = True

  st.session_state.messages.append({"role": "user", "content": prompt})

  with st.chat_message("user"):
    st.markdown(prompt)

  with st.chat_message("assistant"):
    with st.spinner("Searching history & thinking..."):
      try:
        # Step A: Retrieve context from PDF via Retriever if available
        context_text = ""
        if retriever:
          docs_found = retriever.invoke(prompt)
          context_text = "\n\n".join([d.page_content for d in docs_found])

        # Step B: Construct augmented prompt messages for Azure OpenAI client
        augmented_system_prompt = SYSTEM_PROMPT
        if context_text:
          augmented_system_prompt += (
              f"\n\nRETRIEVED DOCUMENT CONTEXT:\n{context_text}"
          )

        # Format history for OpenAI chat completion
        formatted_messages = [{"role": "system", "content": augmented_system_prompt}]
        for m in st.session_state.messages:
          formatted_messages.append(
              {"role": m["role"], "content": m["content"]}
          )

        # Step C: Call Azure OpenAI Client
        client = ChatOpenAI(api_key=API_KEY, base_url=ENDPOINT.rstrip("/") + "/")

        response = client.chat.completions.create(
            model=DEPLOYMENT,
            messages=formatted_messages,
            temperature=0.3,
            max_tokens=800,
        )

        answer = response.choices.message.content
        if not answer:
          answer = (
              "Sorry, I couldn't generate a response. Please try again."
          )

        st.markdown(answer)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

      except Exception as e:
        if st.session_state.messages:
          st.session_state.messages.pop()
        st.error(f"Azure/RAG Error: {e}")

# --------------------------------------------------
# 12. FOOTER
# --------------------------------------------------
st.markdown(
    '<div class="footer-note">Pakistan History AI · Developed by Numan Ali</div>',
    unsafe_allow_html=True,
)