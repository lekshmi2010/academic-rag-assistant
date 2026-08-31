import os
from dotenv import load_dotenv

# Load environment variables from .env file, overriding existing env if updated
load_dotenv(override=True)

# Google Gemini API key for embeddings and chat models
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# Hugging Face token (optional)
HF_TOKEN = os.getenv("HF_TOKEN", "")

# ChromaDB vector store configuration
CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "./chroma_db")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "academic_lectures")

# Models configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-2")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-3.6-flash")

# Session & History Settings
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "12"))
