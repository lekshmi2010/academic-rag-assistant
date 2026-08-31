from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, HttpUrl
import json

# Import updated classification tool alongside your utilities
from services.youtube import extract_video_id, fetch_video_metadata, fetch_full_transcript
from services.classifier import classify_video_content
from services.rag import build_vector_store
from services.chat import ask_lecture_question

app = FastAPI(title="Academic RAG Backend")


class VideoPayload(BaseModel):
    url: HttpUrl

class ChatPayload(BaseModel):
    video_id: str
    question: str
    session_id: str

@app.post("/process_video")
async def process_video(payload: VideoPayload):
    video_url = str(payload.url)
    
    def event_generator():
        try:
            yield json.dumps({"event": "start", "message": "Extracting YouTube video ID..."}) + "\n"
            video_id = extract_video_id(video_url)
            canonical_video_url = f"https://www.youtube.com/watch?v={video_id}"
            
            yield json.dumps({"event": "metadata", "message": f"Fetching metadata for video ID {video_id}..."}) + "\n"
            metadata = fetch_video_metadata(canonical_video_url)
            
            yield json.dumps({"event": "transcript", "message": "Retrieving video transcript..."}) + "\n"
            full_transcript = fetch_full_transcript(video_id)
            
            yield json.dumps({"event": "classify", "message": "Verifying educational content structure via Gemini..."}) + "\n"
            is_approved = classify_video_content(
                metadata["title"],
                full_transcript,
                metadata["categories"],
            )

            if not is_approved:
                yield json.dumps({
                    "event": "error", 
                    "message": "Validation failed. Content parameters do not match a verified academic structure."
                }) + "\n"
                return
                
            yield json.dumps({"event": "rag", "message": "Splitting transcript and compiling chunks into ChromaDB..."}) + "\n"
            total_chunks = build_vector_store(video_id, full_transcript)
            
            yield json.dumps({
                "event": "success",
                "message": "Resource successfully validated and compiled into ChromaDB.",
                "video_id": video_id,
                "processed_chunks": total_chunks
            }) + "\n"

        except HTTPException as http_err:
            yield json.dumps({"event": "error", "message": http_err.detail}) + "\n"
        except ValueError as val_err:
            yield json.dumps({"event": "error", "message": str(val_err)}) + "\n"
        except Exception as e:
            yield json.dumps({"event": "error", "message": f"Unexpected backend failure: {str(e)}"}) + "\n"

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")


@app.post("/chat")
async def chat_with_lecture(payload: ChatPayload):
    """
    Endpoint to ask questions about an ingested lecture.
    """
    try:
        print(f"Searching database for Video ID: {payload.video_id}")
        answer = ask_lecture_question(payload.video_id, payload.question, payload.session_id)

        return {
            "status": "success",
            "video_id": payload.video_id,
            "question": payload.question,
            "session_id": payload.session_id,
            "answer": answer,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
