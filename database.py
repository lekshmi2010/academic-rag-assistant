from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
import config

def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """
    Initializes and returns the Google Generative AI Embeddings.
    """
    return GoogleGenerativeAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        google_api_key=config.GOOGLE_API_KEY
    )

def get_vector_store() -> Chroma:
    """
    Initializes and returns the Chroma Vector Store pointing to the persisted local store.
    """
    return Chroma(
        collection_name=config.CHROMA_COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=config.CHROMA_DB_DIR
    )
