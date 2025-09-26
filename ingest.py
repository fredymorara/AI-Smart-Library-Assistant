import os
import pandas as pd
import wikipedia
import time
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- Basic Setup ---
# This part is still needed to configure the LLM for the main app later.
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_wikipedia_summary(title, author):
    """Attempts to fetch a summary from Wikipedia for a given book."""
    try:
        # Create a more robust search query
        simple_title = title.split('(')[0].strip()
        main_author = author.split('/')[0].strip()
        query = f"{simple_title} ({main_author})"
        
        print(f"  Trying Wikipedia with query: '{query}'")
        # Use auto_suggest=True to handle slight mismatches
        summary = wikipedia.summary(query, sentences=5, auto_suggest=True)
        return summary
    except wikipedia.exceptions.PageError:
        print("  -> Wikipedia PageError: No specific page found.")
        return None
    except wikipedia.exceptions.DisambiguationError as e:
        print(f"  -> Wikipedia DisambiguationError: Found multiple options. Skipping. ({e.options[0]})")
        return None
    except Exception as e:
        print(f"  -> An unexpected error occurred with Wikipedia: {e}")
        return None

def create_enriched_documents():
    """Reads books.csv and enriches it with Wikipedia summaries."""
    df = pd.read_csv('books.csv', on_bad_lines='skip')
    df.dropna(subset=['title', 'authors'], inplace=True)
    
    # Limit to a smaller subset for the first test run, e.g., 50 books.
    # REMOVE or COMMENT OUT the line below for the full run.
    df = df.head(50) 
    
    enriched_docs = []
    for index, row in df.iterrows():
        title = row['title']
        authors = row['authors']
        publisher = row.get('publisher', 'N/A')

        print(f"\nProcessing book {index + 1}/{len(df)}: {title}")
        
        # 1. Start with the basic metadata
        base_info = f"Title: {title}. Authors: {authors}. Publisher: {publisher}."
        
        # 2. Try to get a Wikipedia summary
        summary = get_wikipedia_summary(title, authors)
        
        if summary:
            document_text = f"{base_info} Summary: {summary}"
            print("  -> Successfully enriched with Wikipedia summary.")
        else:
            # Fallback if no summary is found
            document_text = base_info
            print("  -> Storing metadata only.")
            
        enriched_docs.append(document_text)
        
        # 3. Be a polite bot: wait a little between requests to avoid getting blocked
        time.sleep(1) 
        
    return enriched_docs

def ingest_data():
    """Creates the vector database from enriched documents."""
    persist_directory = 'db_enriched'
    
    if os.path.exists(persist_directory):
        print("Enriched vector database already exists. Skipping ingestion.")
        return

    print("Creating enriched document list...")
    documents = create_enriched_documents()
    
    print("\nInitializing local embedding model (this may download the model on first run)...")
    # Use a popular, free, and powerful model from Hugging Face that runs locally
    model_name = "all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    
    print("\nStarting embedding process (this will take a while)...")
    
    db = Chroma.from_texts(
        documents,
        embeddings,
        persist_directory=persist_directory
    )
    
    print(f"\nSuccessfully created enriched vector database with {len(documents)} books.")

if __name__ == "__main__":
    ingest_data()