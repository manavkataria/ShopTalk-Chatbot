import streamlit as st
# import getpass
import json
import os

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# Use environment variable or Streamlit secrets
SECRET_API_KEY = os.environ.get("OPENAI_API_KEY") or st.secrets["OPENAI_API_KEY"]

# Use environment variable or Streamlit secrets
# SECRET_API_KEY = os.environ.get("OPENAI_API_KEY") or st.secrets["OPENAI_API_KEY"]

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = SECRET_API_KEY
# os.environ["LANGSMITH_API_KEY"] = getpass.getpass()

def setup_page_header():
    st.title("💬 ShopTalk Chatbot + LangChain")
    st.write(
        "This is a LangChain chatbot that uses OpenAI's `gpt-4o-mini` model to generate responses. "
        "Provide an OpenAI API key via environment variables or Streamlit secrets. "
    )

def setup_conversation_chain(modelname="gpt-4o-mini"):
    if "conversation" not in st.session_state:
        llm = ChatOpenAI(model=modelname, api_key=SECRET_API_KEY, temperature=0.7)
        memory = ConversationBufferMemory()
        st.session_state.conversation = ConversationChain(llm=llm, memory=memory)

def get_response(prompt):
    return st.session_state.conversation.run(prompt)

def setup_messages():
    if "messages" not in st.session_state:
        st.session_state.messages = []

# Set up
setup_page_header()
setup_conversation_chain()
setup_messages()


# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept new input
if prompt := st.chat_input("What up, Human? 🤖"):
    # Store and display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get bot response
    response = get_response(prompt)
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)

# Debug panel (optional)
with st.expander("Show debug messages"):
    st.code(json.dumps(st.session_state.messages, indent=2), language="json")
