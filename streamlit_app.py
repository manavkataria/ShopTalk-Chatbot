# This is a Streamlit application for an Amazon eCommerce chatbot using LangChain and OpenAI's GPT-4o-mini model.
# The application allows users to interact with the chatbot, which retrieves information from a catalog of Amazon products.
# The chatbot is designed to provide conversational responses based on the product data loaded from a JSON file.
# The application uses ChromaDB for vector storage and retrieval, and it maintains a conversation history using Streamlit's session state.

# Import necessary libraries
import streamlit as st
import json
import os

# Import specific classes
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.callbacks import LangSmithCallbackHandler

# Use environment variable or Streamlit secrets 
SECRET_API_KEY = os.environ.get("OPENAI_API_KEY") or st.secrets["OPENAI_API_KEY"]

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = SECRET_API_KEY

class ChatBotApp:
    def __init__(self, model_name="gpt-4o-mini"):
        """
        Initializes the ChatBotApp class with the given model name.
        """
        self.model_name = model_name
        self.catalog_data = None
        self.vector_store = None

    def load_catalog_data(self, file_path="./data/listings_0_head.json"):
        """
        Loads and parses the Amazon catalog data from the specified JSON file.

        Args:
            file_path (str): Path to the JSON file containing the catalog data.
        """
        with open(file_path, "r") as file:
            data = json.load(file)
            self.catalog_data = [
                {
                    "item_id": item.get("item_id"),
                    "main_image_id": item.get("main_image_id"),
                    "brand": item.get("brand"),
                    "item_keywords": item.get("item_keywords"),
                    "color": item.get("color"),
                    "product_type": item.get("product_type"),
                }
                for item in data
            ]
        st.write("Catalog data loaded successfully.")

    def initialize_chroma_db(self):
        """
        Initializes ChromaDB with the catalog data and creates a vector store for retrieval.

        Raises:
            ValueError: If catalog data is not loaded before initializing the database.
        """
        if not self.catalog_data:
            raise ValueError("Catalog data must be loaded before initializing ChromaDB.")

        embeddings = OpenAIEmbeddings(api_key=SECRET_API_KEY)
        self.vector_store = Chroma.from_documents(
            documents=[
                {
                    "content": f"{item['brand']} {item['product_type']} {item['color']} {item['item_keywords']}",
                    "metadata": item,
                }
                for item in self.catalog_data
            ],
            embedding=embeddings,
        )
        st.write("ChromaDB initialized successfully.")

    def setup_page_header(self):
        """
        Sets up the page header with the title and description of the chatbot.
        Displays the header using Streamlit's markdown with custom HTML.
        """
        st.markdown("<h1 style='text-align: center;'> ShopTalk 💬 Chatbot", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: center;'> Manav Kataria & Artem Strashkov", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: center;'> Amazon eCommerce Chatbot with RAG LLM", unsafe_allow_html=True)
        st.write("Description: This is a LangChain chatbot that uses OpenAI's `gpt-4o-mini` model to generate responses.")

    def setup_langsmith(self):
        """
        Sets up LangSmith tracing for detailed logging and debugging of LangChain operations.
        """
        if "langsmith_handler" not in st.session_state:
            st.session_state.langsmith_handler = LangSmithCallbackHandler()
            st.write("LangSmith tracing initialized.")

    def setup_conversation_chain(self):
        """
        Sets up the conversation chain by initializing the LLM and memory if not already set.
        This function is called only once to avoid overwriting the conversation state with 
        retrieval-augmented generation (RAG) using ChromaDB. 
        """
        if "conversation" not in st.session_state:
            if not self.vector_store:
                raise ValueError("Vector store must be initialized before setting up the conversation chain.")

            llm = ChatOpenAI(model=self.model_name, api_key=SECRET_API_KEY, temperature=0.7)
            retriever = self.vector_store.as_retriever()
            memory = ConversationBufferMemory()
            st.session_state.conversation = ConversationalRetrievalChain(
                retriever=retriever, llm=llm, memory=memory, callbacks=[st.session_state.langsmith_handler]
            )

    def setup_messages(self):
        """
        Sets up the messages in session state if they are not already initialized.
        Ensures that the chatbot's chat history is stored and persists across reruns.
        """
        if "messages" not in st.session_state:
            st.session_state.messages = []

    def run_setup_once(self):
        """
        Runs the setup process only once, ensuring that page header, conversation chain,
        LangSmith tracing, and message history are initialized when needed.
        This prevents setup from running on every rerun.
        """
        if not st.session_state.get("initialized"):
            self.setup_langsmith()  # Initialize LangSmith tracing
            self.load_catalog_data()  # Load catalog data
            self.initialize_chroma_db()  # Initialize ChromaDB
            self.setup_conversation_chain()  # Setup conversation chain
            self.setup_messages()
            st.session_state.initialized = True

    def get_response(self, prompt):
        """
        Gets the response from the chatbot for the given prompt by running the conversation chain.
        
        Args:
            prompt (str): The user's message to send to the chatbot.

        Returns:
            str: The response generated by the chatbot.
        """
        return st.session_state.conversation.run(prompt)

    def handle_user_input(self):
        """
        Handles the user input by appending it to the session's message history and generating
        a response using the chatbot. Also displays the user's input and the bot's response in the chat.
        """
        if prompt := st.chat_input("What up, Human? 🤖"):
            # Store and render user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Get bot response and render. #TODO: See if we can merge the rendering logic 
            # in a single render_message_history() --> render_messages()
            response = self.get_response(prompt)
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)

    def render_message_history(self):
        """
        Renders all the messages stored in session state. This displays the entire chat history
        between the user and the assistant.
        """
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    def show_debug(self):
        """
        Displays debug information for developers, showing the message history in JSON format
        inside an expandable section. This helps in inspecting the session state.
        Additionally, displays the catalog data if loaded.
        """
        with st.expander("Show debug messages"):
            st.code(json.dumps(st.session_state.messages, indent=2), language="json")
        
        with st.expander("Show catalog data"):
            st.code(json.dumps(self.catalog_data if self.catalog_data else "{'self.catalog_data is empty'}", indent=2), language="json")
            

    def render_app(self):
        """
        Renders the chatbot application by initializing the setup, displaying chat messages,
        handling new user input, and showing debug information if needed.
        """
        self.run_setup_once()       # Run this setup logic only once
        self.setup_page_header()    # Always render header on each rerun
        self.render_message_history()
        self.handle_user_input()
        self.show_debug()


if __name__ == "__main__":
    app = ChatBotApp()
    app.render_app()
