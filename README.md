# Academic RAG Assistant

A FastAPI + Streamlit application that ingests YouTube lecture videos, validates academic content using LLM classification, builds a RAG (Retrieval-Augmented Generation) pipeline with ChromaDB and Gemini embeddings, and provides an interactive chat interface for Q&A over lecture transcripts.

![Demo](https://img.shields.io/badge/demo-available-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/python-3.11+-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.136-009688?style=for-the-badge&logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-1.47-FF4B4B?style=for-the-badge&logo=streamlit)

## Overview

This application transforms YouTube educational content into an interactive learning experience. Simply provide a YouTube URL of an academic lecture, and the system will:
1. Extract and validate the video content as legitimate educational material
2. Process the transcript into a searchable vector database
3. Enable natural language Q&A with the lecture content, complete with AI reasoning display

## Features

- **YouTube Video Ingestion**: Extract video ID, fetch metadata, and retrieve full transcripts via `youtube-transcript-api` and `yt-dlp`
- **Academic Content Validation**: Uses Gemini to classify whether a video has genuine educational/academic structure (filters out vlogs, entertainment, gaming content)
- **RAG Pipeline**: Splits transcripts into chunks (1000 chars with 200 overlap), generates embeddings with `gemini-embedding-2`, stores in ChromaDB with video-level filtering
- **Interactive Chat**: Streamlit frontend with session-based conversation history, streaming status updates, and thinking/reasoning display
- **FastAPI Backend**: RESTful API with SSE streaming for video processing, standard JSON for chat

### AI Reasoning Display

The chat interface shows the model's step-by-step reasoning process, helping you understand how answers are derived from the lecture content.

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   YouTube URL   │────▶│  FastAPI Backend│────▶│   ChromaDB      │
│   (User Input)  │     │  (main.py)      │     │   Vector Store  │
└─────────────────┘     └────────┬────────┘     └─────────────────┘
                                 │
                    ┌────────────┼────────────┐
                    ▼            ▼            ▼
             ┌──────────┐ ┌──────────┐ ┌──────────┐
             │ YouTube  │ │ Classifier│ │   RAG    │
             │ Services │ │ (Gemini)  │ │ Pipeline │
             └──────────┘ └──────────┘ └──────────┘
                    │
                    ▼
             ┌─────────────────┐
             │ Streamlit Frontend│
             │ (app.py)          │
             └─────────────────┘
```

## Project Structure

```
youtube-podcast-chatbot/
├── main.py              # FastAPI application entry point
├── app.py               # Streamlit frontend
├── config.py            # Configuration (API keys, model names, DB paths)
├── requirements.txt     # Python dependencies
├── database.py          # ChromaDB connection management
├── services/
│   ├── youtube.py       # Video ID extraction, metadata, transcripts
│   ├── classifier.py    # Academic content validation via Gemini
│   ├── rag.py           # Vector store building & ingestion
│   └── chat.py          # RAG-based Q&A with session history
└── utils/
    └── text.py          # Text normalization utilities
```

## Prerequisites

- Python 3.11+
- Google Gemini API key
- Hugging Face token (optional, for some embeddings)

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd youtube-podcast-chatbot

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Edit `config.py` with your API keys:

```python
HF_TOKEN = "your_huggingface_token"
GOOGLE_API_KEY = "your_google_gemini_api_key"

CHROMA_DB_DIR = "./chroma_db"
CHROMA_COLLECTION_NAME = "academic_lectures"
EMBEDDING_MODEL = "gemini-embedding-2"
CHAT_MODEL = "gemini-3.1-flash-lite"
MAX_HISTORY_MESSAGES = 12
```

## Running the Application

### 1. Start the FastAPI Backend

```bash
uvicorn main:app --reload --port 8000
```

The API will be available at `http://127.0.0.1:8000`

### 2. Start the Streamlit Frontend

```bash
streamlit run app.py
```

The UI will open at `http://localhost:8501`

## API Endpoints

### `POST /process_video`
Process a YouTube video URL through the full pipeline.

**Request:**
```json
{
  "url": "https://www.youtube.com/watch?v=VIDEO_ID"
}
```

**Response (SSE stream):**
```
{"event": "start", "message": "Extracting YouTube video ID..."}
{"event": "metadata", "message": "Fetching metadata for video ID ..."}
{"event": "transcript", "message": "Retrieving video transcript..."}
{"event": "classify", "message": "Verifying educational content structure via Gemini..."}
{"event": "rag", "message": "Splitting transcript and compiling chunks into ChromaDB..."}
{"event": "success", "message": "Resource successfully validated...", "video_id": "...", "processed_chunks": 42}
```

### `POST /chat`
Ask a question about an ingested lecture.

**Request:**
```json
{
  "video_id": "VIDEO_ID",
  "question": "What is the main topic of this lecture?",
  "session_id": "unique-session-id"
}
```

**Response:**
```json
{
  "status": "success",
  "video_id": "VIDEO_ID",
  "question": "What is the main topic of this lecture?",
  "session_id": "unique-session-id",
  "answer": "Based on the lecture, the main topic is..."
}
```

## Usage

1. Open the Streamlit UI at `http://localhost:8501`
2. Paste a YouTube lecture URL (e.g., `https://www.youtube.com/watch?v=zjkBMFhNj_g`)
3. Click "Process & Load Lecture" - watch the streaming status
4. Once loaded, the video player and chat interface appear
5. Ask questions about the lecture content

### Pre-verified Test Lectures

The UI includes quick-load buttons for:
- Andrej Karpathy - "Let's build GPT" (LLMs)
- 3Blue1Brown - Linear Algebra
- MIT 6.006 - Introduction to Algorithms

## Key Technologies

| Technology | Purpose | Version |
|------------|---------|---------|
| **FastAPI** | High-performance async API framework | `0.136+` |
| **Streamlit** | Rapid data app frontend | `1.47+` |
| **LangChain** | RAG orchestration (chroma, text-splitters, google-genai) | `1.4+` |
| **ChromaDB** | Local vector database with persistence | `1.1+` |
| **Google Gemini** | Embeddings (`gemini-embedding-2`) and chat (`gemini-3.1-flash-lite`) | Latest |
| **youtube-transcript-api** | Transcript extraction | `1.2+` |
| **yt-dlp** | Metadata fetching | `2026+` |
| **Pydantic** | Request validation and data models | `2.13+` |

### Text Processing

- Uses `RecursiveCharacterTextSplitter` for intelligent chunking
- Chunk size: 1000 characters with 200 character overlap
- Separators prioritize paragraph breaks, then sentences, for coherent context preservation

## RAG Evaluation & Benchmarking (Ragas)

The system includes an automated evaluation suite built with **[Ragas](https://github.com/explodinggradients/ragas)** to measure retrieval quality, factual grounding, and response accuracy across multi-modal video transcripts.

### Evaluated Metrics

- **Faithfulness** (Hallucination Detection): Measures whether the generated answer contains only claims that can be inferred from the retrieved transcript context.
- **Answer Relevancy**: Evaluates how pertinent the model's response is to the user's specific query.
- **Context Precision**: Determines whether ground-truth relevant chunks appear at the highest ranks of retrieved documents.
- **Context Recall**: Verifies that the retrieved chunks contain all necessary knowledge required to answer the ground-truth reference.
- **Vector Retrieval Latency**: Measures ChromaDB top-$k$ similarity search execution speed in milliseconds.
- **End-to-End Latency**: Total roundtrip time from question to reasoning output.

### Running the Evaluation Suite

```bash
python evaluate.py
```

Results are automatically logged to the terminal and exported to:
- `ragas_benchmark_results.csv` (detailed per-query metrics)
- `ragas_benchmark_summary.md` (formatted scorecard for portfolio and documentation)

## Error Handling

## Error Handling

The frontend provides contextual error messages for:
- **Academic validation failure** - Video doesn't meet educational structure criteria
- **Missing transcripts** - Captions disabled or unavailable
- **Connection errors** - Backend not running or unreachable
- **Invalid URL** - Cannot parse YouTube video ID

## License

MIT License