# rag_pipeline.py
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import os

def create_vector_store(text_content: str):
    """
    Chunks textual data and generates a FAISS vector store instance.
    """
    
    openai_embed_key = os.environ.get("OPEN_AI_EMBED_KEY", "")
    openai_embed_url = os.environ.get("OPEN_AI_EMBED_ENDPOINT", "")  
    
    # Initialize the target embedding configuration using the global ecosystem variables
    embeddings = OpenAIEmbeddings(
        api_key = openai_embed_key,
        base_url = openai_embed_url,
        model = 'text-embedding-3-large'
    )

    if not text_content.strip():
        return None
        
    # Split manual into reasonable procedural chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    docs = text_splitter.create_documents([text_content])

    
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store

def retrieve_docs(vector_store, query: str, k: int = 3):
    """
    Extracts top matching knowledge frames for grounding support queries.
    """
    if vector_store is None:
        return []
    return vector_store.similarity_search(query, k=k)