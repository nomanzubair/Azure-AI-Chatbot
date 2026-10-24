
import os
import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="AI Chatbot", page_icon="🤖")
st.title("🤖 AI Chatbot")

API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

if not API_KEY or not ENDPOINT or not DEPLOYMENT:
    st.error(
        "Azure OpenAI ki environment settings missing hain. "
        "Azure App Settings check karein."
    )
    st.stop()

client = OpenAI(
    api_key=API_KEY,
    base_url=ENDPOINT.rstrip("/") + "/"
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Apna sawal yahan likhein..."):
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            response = client.chat.completions.create(
                model=DEPLOYMENT,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful AI assistant."
                    },
                    *st.session_state.messages
                ],
            )

            answer = response.choices[0].message.content

            if not answer:
                answer = "AI se empty response mila hai."

            st.markdown(answer)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )

        except Exception as e:
            st.error(f"Azure OpenAI Error: {e}")

            # Failed request ko chat history se hata dein
            st.session_state.messages.pop()
