from langchain_text_splitters import RecursiveCharacterTextSplitter
from database import get_vector_store
import config

def build_vector_store(video_id: str, full_transcript: str) -> int:
    """
    Splits the full transcript into semantically cohesive chunks,
    generates embeddings using Gemini, and saves them to ChromaDB.
    """
    if not config.GOOGLE_API_KEY or config.GOOGLE_API_KEY == "your_gemini_api_key_here":
        print("[RAG Error] GOOGLE_API_KEY is missing. Cannot vectorize text.")
        return 0

    if not full_transcript.strip():
        print("[RAG Warning] Transcript is empty. Skipping database ingestion.")
        return 0

    try:
        # 1. Initialize Text Splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

        # 2. Divide text and inject structural metadata
        chunks = text_splitter.create_documents(
            texts=[full_transcript],
            metadatas=[{"video_id": video_id}]
        )
        
        total_chunks = len(chunks)
        print(f"[RAG System] Created {total_chunks} text chunks from transcript.")

        # 3. Retrieve persistent Vector Database connection
        vector_store = get_vector_store()
        
        # 4. Ingest the documents into the database
        vector_store.add_documents(chunks)
        print(f"[RAG System] Successfully ingested chunks into local store at: {config.CHROMA_DB_DIR}")
        
        return total_chunks

    except Exception as e:
        print(f"[RAG Error] Vector store generation failed: {e}")
        return 0
