import os
from openai import AzureOpenAI
import streamlit as st

# Page configuration
st.set_page_config(page_title="AI Chatbot", page_icon="🤖")
st.title("🤖 AI Chatbot")

# Azure OpenAI credentials from environment variables
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

if not AZURE_OPENAI_API_KEY or not AZURE_OPENAI_ENDPOINT:
  st.warning(
      "Kripya apni environment variables (.env file ya Azure App Settings) mein"
      " API key aur endpoint set karein."
  )
else:
  client = AzureOpenAI(
      api_key=AZURE_OPENAI_API_KEY,
      api_version="2024-02-01",
      azure_endpoint=AZURE_OPENAI_ENDPOINT,
  )

  # Initialize chat history
  if "messages" not in st.session_state:
    st.session_state.messages = []

  # Display chat history
  for message in st.session_state.messages:
    with st.chat_message(message["role"]):
      st.markdown(message["content"])

  # User input
  if prompt := st.chat_input("Apna sawal yahan likhein..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
      st.markdown(prompt)

    with st.chat_message("assistant"):
      try:
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
        )
        answer = response.choices[0].message.content
        st.markdown(answer)
        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )
      except Exception as e:
        st.error(f"Koi error aa gaya hai: {e}")
