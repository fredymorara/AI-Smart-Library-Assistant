import os
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import GooglePalm # Using the legacy PaLM model for simplicity if Gemini fails

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Smart Library Assistant",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="auto",
)

# --- LOAD ENVIRONMENT VARIABLES AND API KEY ---
load_dotenv()
# Note: The genai.configure call is now done inside the function where the LLM is used.

# --- LOAD THE VECTOR DATABASE ---
# This is the "brain" you created with ingest.py
# We use a cached function so it only loads the model and DB once.
@st.cache_resource
def load_retriever():
    print("Loading vector database and embedding model...")
    persist_directory = 'db_enriched'
    
    # Use the same local embedding model
    model_name = "all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    
    # Load the existing vector store
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    
    # Expose the vector store as a retriever
    # k=3 means it will find the 3 most relevant book chunks
    retriever = db.as_retriever(search_kwargs={"k": 3})
    print("Loading complete.")
    return retriever

# --- DEFINE THE PROMPT TEMPLATE ---
prompt_template = """You are a helpful and enthusiastic Smart Library Assistant for Kabarak University.
Your goal is to help users discover books from the library's collection based on their questions.
Use the following pieces of context from the library catalogue to answer the user's question.
If you don't know the answer from the context provided, just say that you don't know, don't try to make up an answer.
Provide the titles of the books you are recommending.

CONTEXT:
{context}

QUESTION:
{question}

HELPFUL ANSWER:
"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

# --- GENERATE RESPONSE ---
def generate_response(retriever, question):
    """Generates a response to the user's question using the RAG pipeline."""
    try:
        # 1. Retrieve relevant documents from the vector store
        docs = retriever.get_relevant_documents(question)
        
        # Check if any documents were found
        if not docs:
            return "I'm sorry, I couldn't find any books in our collection that match your query. Could you try asking in a different way?"
        
        # Format the context for the prompt
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # 2. Configure the Google LLM
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        llm = genai.GenerativeModel('gemini-1.5-flash-latest')
        
        # 3. Create the prompt with the retrieved context
        formatted_prompt = PROMPT.format(context=context, question=question)

        # 4. Generate the response from the LLM
        response = llm.generate_content(formatted_prompt)
        
        return response.text

    except Exception as e:
        # A general error message to catch any API or other issues
        st.error(f"An error occurred: {e}")
        return "I'm having a little trouble right now. Please try again in a moment."

# --- STREAMLIT APP LAYOUT ---
st.title("📚 Smart Library Assistant")
st.caption("Your personal guide to the Kabarak University library collection.")

# Load the retriever once
retriever = load_retriever()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you find a book today?"}]

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask me about a book, topic, or author..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        with st.spinner("Searching the library..."):
            response = generate_response(retriever, prompt)
            st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})