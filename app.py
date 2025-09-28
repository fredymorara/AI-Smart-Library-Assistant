import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_groq import ChatGroq  # The new, simple import

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Smart Library Assistant",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="auto",
)

 # --- LOAD ENVIRONMENT VARIABLES ---
 # Only load the .env file if it exists (i.e., we're running locally)
 if os.path.exists('.env'):
     from dotenv import load_dotenv
     load_dotenv()


# --- LOAD THE VECTOR DATABASE ---
@st.cache_resource
def load_retriever():
    print("Loading vector database and embedding model...")
    persist_directory = 'db_enriched'
    
    model_name = "all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    
    retriever = db.as_retriever(search_kwargs={"k": 5}) # Increased to 5 for more context
    print("Loading complete.")
    return retriever

# --- DEFINE THE PROMPT TEMPLATE ---
prompt_template = """You are a helpful and enthusiastic Smart Library Assistant for Kabarak University.
Your goal is to help users discover books from the library's collection based on their questions.
Use the following pieces of context from the library catalogue to answer the user's question.
If you don't know the answer from the context provided, just say that you don't have enough information in the catalogue to answer.
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
        docs = retriever.get_relevant_documents(question)
        
        if not docs:
            return "I'm sorry, I couldn't find any books in our collection that match your query. Could you try asking in a different way?"
        
        # Initialize the Groq Chat LLM
        # Llama3 is one of the best open-source models available.
        llm = ChatGroq(
            temperature=0.2,
            model_name="llama-3.1-8b-instant",
            api_key=os.getenv("GROQ_API_KEY") # Pass the key directly
        )

        chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=PROMPT)
        
        response = chain.invoke({"input_documents": docs, "question": question})
        
        return response.get('output_text', "Sorry, I had trouble generating a response.")

    except Exception as e:
        st.error(f"An error occurred: {e}")
        return "I'm having a little trouble connecting to the AI service. Please check your API key and try again."

# --- STREAMLIT APP LAYOUT ---
st.title("📚 Smart Library Assistant")
st.caption("Your personal guide to the Kabarak University library collection.")

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
        with st.spinner("Searching the library... (This will be fast!)"):
            response = generate_response(retriever, prompt)
            st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})