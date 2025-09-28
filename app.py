import os
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_groq import ChatGroq

# --- HYBRID API KEY MANAGEMENT ---
# This is the robust, environment-aware logic you proposed.

# Check if we are running on Streamlit Cloud. The presence of this secrets key is a reliable indicator.
if 'GROQ_API_KEY' in st.secrets:
    # We are in the cloud, use Streamlit Secrets
    print("Running on Streamlit Cloud. Using st.secrets.")
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
else:
    # We are running locally, use the .env file
    print("Running locally. Loading from .env file.")
    from dotenv import load_dotenv
    load_dotenv()
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# A crucial final check to ensure the key was loaded from one of the sources
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found! Please set it in your .env file locally or in your Streamlit secrets.")
    st.stop()

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Smart Library Assistant",
    page_icon="📚",
    layout="centered",
)

# --- LOAD THE VECTOR DATABASE ---
@st.cache_resource
def load_retriever():
    print("Loading vector database and embedding model...")
    persist_directory = 'db_enriched'
    model_name = "all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 5})
    print("Loading complete.")
    return retriever

# --- DEFINE THE PROMPT TEMPLATE ---
prompt_template = """You are a helpful and enthusiastic Smart Library Assistant.
Use the following pieces of context from the library catalogue to answer the user's question.
If you don't know the answer from the context provided, just say that you don't have enough information to answer.
Provide the titles of the books you are recommending.

CONTEXT: {context}
QUESTION: {question}
HELPFUL ANSWER:"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# --- GENERATE RESPONSE ---
def generate_response(retriever, question):
    try:
        docs = retriever.get_relevant_documents(question)
        if not docs:
            return "I'm sorry, I couldn't find any books that match your query. Could you try asking in a different way?"
        
        llm = ChatGroq(
            temperature=0.2,
            model_name="llama-3.1-8b-instant", # Using the known-good model
            api_key=GROQ_API_KEY # Use the single, reliable key variable
        )
        chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=PROMPT)
        response = chain.invoke({"input_documents": docs, "question": question})
        return response.get('output_text', "Sorry, I had trouble generating a response.")

    except Exception as e:
        st.error(f"An error occurred: {e}")
        return "I'm having a little trouble connecting to the AI service. Please check your API key and try again."

# --- STREAMLIT APP LAYOUT ---
st.title("📚 Smart Library Assistant")
st.caption("Your personal guide to the library collection.")

retriever = load_retriever()

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you find a book today?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me about a book, topic, or author..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching the library..."):
            response = generate_response(retriever, prompt)
            st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})