# import os
# import streamlit as st
# from openai import OpenAI

# # Page Configuration and Custom UI Styling
# st.set_page_config(
#     page_title="Professional AI Assistant",
#     page_icon="🤖",
#     layout="centered",
#     initial_sidebar_state="expanded"
# )

# # Custom CSS for enhanced UI styling
# st.markdown("""
#     <style>
#     .main {
#         background-color: #f8f9fa;
#     }
#     .stChatInputContainer {
#         padding-bottom: 20px;
#     }
#     div[data-testid="stChatMessage"] {
#         padding: 1rem;
#         border-radius: 12px;
#         margin-bottom: 1rem;
#         box-shadow: 0 1px 3px rgba(0,0,0,0.05);
#     }
#     </style>
# """, unsafe_allow_html=True)

# # Sidebar Configuration for UI layout
# with st.sidebar:
#     st.image("https://img.icons8.com/color/96/artificial-intelligence.png", width=80)
#     st.title("Control Panel")
#     st.markdown("---")
#     st.markdown("### About")
#     st.info(
#         "This is a professional AI-powered assistant built with Streamlit and Azure OpenAI. "
#         "Type your queries below to start the conversation."
#     )
#     if st.button("Clear Conversation History", type="secondary"):
#         st.session_state.messages = []
#         st.rerun()

# # Main Header Section
# st.title("🤖 Enterprise AI Assistant")
# st.caption("Your intelligent workspace companion powered by Azure OpenAI.")
# st.markdown("---")

# API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
# ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
# DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

# if not API_KEY or not ENDPOINT or not DEPLOYMENT:
#     st.error(
#         "Azure OpenAI environment settings are missing. "
#         "Please check your Azure App Settings or configuration file."
#     )
#     st.stop()

# client = OpenAI(
#     api_key=API_KEY,
#     base_url=ENDPOINT.rstrip("/") + "/"
# )

# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # Display chat history with enhanced UI layout
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# if prompt := st.chat_input("Type your question here..."):
#     st.session_state.messages.append(
#         {"role": "user", "content": prompt}
#     )

#     with st.chat_message("user"):
#         st.markdown(prompt)

#     with st.chat_message("assistant"):
#         with st.spinner("Thinking..."):
#             try:
#                 response = client.chat.completions.create(
#                     model=DEPLOYMENT,
#                     messages=[
#                         {
#                             "role": "system",
#                             "content": "You are a helpful AI assistant."
#                         },
#                         *st.session_state.messages
#                     ],
#                 )

#                 answer = response.choices[0].message.content

#                 if not answer:
#                     answer = "Received an empty response from the AI."

#                 st.markdown(answer)
#                 st.session_state.messages.append(
#                     {"role": "assistant", "content": answer}
#                 )

#             except Exception as e:
#                 st.error(f"Azure OpenAI Error: {e}")

#                 # Remove the failed request from chat history
#                 st.session_state.messages.pop()





# import os
# import streamlit as st
# from openai import OpenAI

# st.set_page_config(page_title="AI Chatbot", page_icon="🤖")
# st.title("🤖 AI Chatbot")

# API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
# ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
# DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

# if not API_KEY or not ENDPOINT or not DEPLOYMENT:
#     st.error(
#         "Azure OpenAI ki environment settings missing hain. "
#         "Azure App Settings check karein."
#     )
#     st.stop()

# client = OpenAI(
#     api_key=API_KEY,
#     base_url=ENDPOINT.rstrip("/") + "/"
# )

# if "messages" not in st.session_state:
#     st.session_state.messages = []

# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# if prompt := st.chat_input("Apna sawal yahan likhein..."):
#     st.session_state.messages.append(
#         {"role": "user", "content": prompt}
#     )

#     with st.chat_message("user"):
#         st.markdown(prompt)

#     with st.chat_message("assistant"):
#         try:
#             response = client.chat.completions.create(
#                 model=DEPLOYMENT,
#                 messages=[
#                     {
#                         "role": "system",
#                         "content": "You are a helpful AI assistant."
#                     },
#                     *st.session_state.messages
#                 ],
#             )

#             answer = response.choices[0].message.content

#             if not answer:
#                 answer = "AI se empty response mila hai."

#             st.markdown(answer)
#             st.session_state.messages.append(
#                 {"role": "assistant", "content": answer}
#             )

#         except Exception as e:
#             st.error(f"Azure OpenAI Error: {e}")

#             # Failed request ko chat history se hata dein
#             st.session_state.messages.pop()
