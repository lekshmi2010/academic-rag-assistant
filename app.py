import uuid
import json
import requests
import streamlit as st

DEFAULT_API_URL = "http://127.0.0.1:8000"
REQUEST_TIMEOUT = 180

st.set_page_config(page_title="Academic RAG Assistant", page_icon="📚", layout="centered")

from utils.text import normalize_answer_content

def initialize_state() -> None:
    if "api_url" not in st.session_state:
        st.session_state.api_url = DEFAULT_API_URL
    if "youtube_url" not in st.session_state:
        st.session_state.youtube_url = ""
    if "video_id" not in st.session_state:
        st.session_state.video_id = None
    if "processed" not in st.session_state:
        st.session_state.processed = False
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid.uuid4().hex
    if "error_message" not in st.session_state:
        st.session_state.error_message = None

def reset_chat_for_new_video(video_id: str) -> None:
    st.session_state.video_id = video_id
    st.session_state.processed = True
    st.session_state.messages = []
    st.session_state.session_id = uuid.uuid4().hex

def submit_question(prompt: str) -> None:
    if not st.session_state.video_id:
        st.warning("Process a lecture before asking questions.")
        return

    st.session_state.messages.append({"role": "user", "content": prompt})

    try:
        with st.spinner("Thinking..."):
            response = requests.post(
                f"{st.session_state.api_url.rstrip('/')}/chat",
                json={
                    "video_id": st.session_state.video_id,
                    "question": prompt,
                    "session_id": st.session_state.session_id,
                },
                timeout=REQUEST_TIMEOUT,
            )

        if response.ok:
            raw_answer = normalize_answer_content(response.json().get("answer", ""))

            # Try to parse the assistant's response as JSON per the updated prompt
            try:
                parsed = json.loads(raw_answer)
                final = parsed.get("final_answer")
                thinking = parsed.get("thinking")

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "final_answer": final or "",
                        "thinking": thinking if isinstance(thinking, list) else None,
                    }
                )
            except Exception:
                st.session_state.messages.append({"role": "assistant", "final_answer": raw_answer, "thinking": None})
        else:
            detail = response.json().get("detail", "Backend error while generating response.")
            st.session_state.messages.append({"role": "assistant", "final_answer": f"Error: {detail}", "thinking": None})
    except requests.exceptions.ConnectionError:
        st.session_state.messages.append(
            {"role": "assistant", "final_answer": "Connection error. Is the FastAPI backend running?", "thinking": None}
        )
    except requests.exceptions.Timeout:
        st.session_state.messages.append(
            {"role": "assistant", "final_answer": "The backend timed out while generating the answer.", "thinking": None}
        )
    except Exception as exc:
        st.session_state.messages.append({"role": "assistant", "final_answer": f"Unexpected error: {exc}", "thinking": None})

initialize_state()

# Inject premium styling
st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 900px;
    }
    html, body, [class*="css"], .stApp {
        font-family: 'Outfit', sans-serif;
    }
    .hero {
        padding: 2rem;
        border-radius: 1.25rem;
        background: linear-gradient(135deg, rgba(17, 24, 39, 0.95), rgba(31, 41, 55, 0.9));
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
        margin-bottom: 2rem;
        text-align: center;
    }
    .hero h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(45deg, #3b82f6, #8b5cf6, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero p {
        margin: 0.75rem 0 0;
        opacity: 0.85;
        font-size: 1.05rem;
        color: #d1d5db;
    }
    .panel {
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 1.25rem;
        padding: 1.5rem;
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(12px);
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.05);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>Academic RAG Assistant</h1>
        <p>Analyze academic YouTube lectures, verify material, and engage in interactive Q&A directly below.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar configuration
with st.sidebar:
    st.subheader("⚙️ Backend Settings")
    st.session_state.api_url = st.text_input("FastAPI URL", value=st.session_state.api_url)
    st.caption("Default: http://127.0.0.1:8000")

    st.divider()
    st.subheader("🔑 Active Session")
    st.code(st.session_state.session_id, language="text")
    if st.session_state.video_id:
        st.success(f"Video Loaded: {st.session_state.video_id}")
        if st.button("Reset App", use_container_width=True):
            st.session_state.video_id = None
            st.session_state.processed = False
            st.session_state.messages = []
            st.session_state.youtube_url = ""
            st.rerun()
    else:
        st.info("No active lecture.")

# Main input container for URL submission
st.markdown('<div class="panel">', unsafe_allow_html=True)
st.subheader("🔗 Ingest Academic YouTube Video")
youtube_url = st.text_input(
    "Paste YouTube URL here:",
    value=st.session_state.youtube_url,
    placeholder="https://www.youtube.com/watch?v=...",
    label_visibility="collapsed"
)
st.session_state.youtube_url = youtube_url

process_clicked = st.button("🚀 Process & Load Lecture", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# Run process video when clicked
if process_clicked:
    if not youtube_url.strip():
        st.warning("Please paste a valid YouTube URL first.")
    else:
        st.session_state.error_message = None  # Clear previous errors
        # Live streaming status block
        with st.status("Initializing Ingestion...", expanded=True) as status_box:
            try:
                response = requests.post(
                    f"{st.session_state.api_url.rstrip('/')}/process_video",
                    json={"url": youtube_url},
                    stream=True,
                    timeout=REQUEST_TIMEOUT,
                )
                
                success = False
                video_id = None
                processed_chunks = 0
                error_message = None
                
                if response.status_code == 200:
                    for line in response.iter_lines():
                        if line:
                            data = json.loads(line.decode('utf-8'))
                            event = data.get("event")
                            msg = data.get("message")
                            
                            if event == "error":
                                status_box.update(label="Ingestion failed!", state="error")
                                error_message = msg
                                break
                            elif event == "success":
                                success = True
                                video_id = data.get("video_id")
                                processed_chunks = data.get("processed_chunks")
                                status_box.update(label="Ingestion completed successfully!", state="complete")
                                break
                            else:
                                # Render the current step
                                st.write(f"⏳ {msg}")
                else:
                    try:
                        detail = response.json().get("detail", "Backend failure.")
                    except Exception:
                        detail = f"HTTP error {response.status_code}"
                    status_box.update(label="Ingestion failed!", state="error")
                    error_message = detail
                
                if success and video_id:
                    st.session_state.error_message = None
                    reset_chat_for_new_video(video_id)
                    st.success(f"Successfully loaded video ID: {video_id} ({processed_chunks} chunks).")
                    st.rerun()
                elif error_message:
                    st.session_state.error_message = error_message
                    st.rerun()
                    
            except requests.exceptions.ConnectionError:
                status_box.update(label="Failed to reach backend", state="error")
                st.session_state.error_message = "Cannot connect to the backend. Is the FastAPI service running?"
                st.rerun()
            except Exception as e:
                status_box.update(label="Unexpected error occurred", state="error")
                st.session_state.error_message = f"Unexpected error: {e}"
                st.rerun()

# Ingestion error panel and alternative content loader
if st.session_state.error_message:
    st.markdown('<div class="panel" style="border: 1px solid rgba(239, 68, 68, 0.4); background: rgba(239, 68, 68, 0.05);">', unsafe_allow_html=True)
    st.subheader("⚠️ Ingestion Failure")
    
    err = st.session_state.error_message.lower()
    
    if "validation failed" in err or "academic structure" in err:
        st.error(
            "**What is the problem?**\n\n"
            "This video does not appear to be structured educational or academic content. "
            "Our routing system filters out casual vlogs, entertainment, gaming, and general talk shows to keep the QA focused.\n\n"
            "**What can you do?**\n\n"
            "Please provide a link to a formal lecture, computer science tutorial, or mathematical derivation. "
            "Alternatively, click one of the pre-verified academic lectures below to test the assistant immediately!"
        )
    elif "disabled" in err or "captions" in err or "transcript" in err or "no english" in err:
        st.error(
            "**What is the problem?**\n\n"
            "English transcripts/captions are disabled or unavailable for this video. "
            "Our assistant requires text captions to partition and search the lecture content.\n\n"
            "**What can you do?**\n\n"
            "Please choose a YouTube video that has English subtitles or auto-generated captions enabled. "
            "You can verify this on YouTube by ensuring the 'CC' button is visible and can be toggled to English."
        )
    elif "connection" in err or "connect" in err:
        st.error(
            "**What is the problem?**\n\n"
            "We could not establish a connection to the backend service.\n\n"
            "**What can you do?**\n\n"
            "Make sure the backend server is running (usually by executing `uvicorn main:app --reload` in your terminal) "
            "and that the URL configured in the sidebar matches your running service."
        )
    elif "extract" in err or "video id" in err:
        st.error(
            "**What is the problem?**\n\n"
            "The system was unable to parse a valid YouTube Video ID from the link you pasted.\n\n"
            "**What can you do?**\n\n"
            "Verify the link format. Make sure you copy the complete link from your web browser's address bar "
            "(e.g., `https://www.youtube.com/watch?v=...` or `https://youtu.be/...`)."
        )
    else:
        st.error(
            "**What is the problem?**\n\n"
            "An unexpected backend failure occurred during video processing.\n\n"
            "**What can you do?**\n\n"
            "Please check the server console logs for details, try clicking 'Clear URL & Retry', or load one of our verified academic lectures below."
        )
        
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🧹 Clear URL & Retry", use_container_width=True):
            st.session_state.youtube_url = ""
            st.session_state.error_message = None
            st.rerun()
            
    st.markdown("---")
    st.write("💡 **Try loading one of these verified academic lectures:**")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Andrej Karpathy (Build GPT)", use_container_width=True):
            st.session_state.youtube_url = "https://www.youtube.com/watch?v=zjkBMFhNj_g"
            st.session_state.error_message = None
            st.rerun()
    with c2:
        if st.button("3Blue1Brown (Neural Networks)", use_container_width=True):
            st.session_state.youtube_url = "https://www.youtube.com/watch?v=aircAruvnKk"
            st.session_state.error_message = None
            st.rerun()
    with c3:
        if st.button("Karpathy (Intro to LLMs)", use_container_width=True):
            st.session_state.youtube_url = "https://www.youtube.com/watch?v=kCc8FmEb1nY"
            st.session_state.error_message = None
            st.rerun()
            
    st.markdown("</div>", unsafe_allow_html=True)

# If video is loaded, show stacked player and chatbot
if st.session_state.video_id:
    # 1. Video Player Section
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("📺 Lecture Video")
    st.video(f"https://www.youtube.com/watch?v={st.session_state.video_id}")
    st.markdown('</div>', unsafe_allow_html=True)

    # 2. Chat Interface Section
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("💬 Chat with the Lecture")

    if st.session_state.messages:
        for message in st.session_state.messages:
            if message.get("role") == "user":
                with st.chat_message("user"):
                    st.markdown(message.get("content", ""))
                continue

            with st.chat_message("assistant"):
                final = message.get("final_answer") or message.get("content", "")
                st.markdown(final)

                thinking = message.get("thinking")
                if thinking and isinstance(thinking, list):
                    with st.expander("Show reasoning"):
                        for i, step in enumerate(thinking, start=1):
                            if isinstance(step, dict):
                                text = step.get("text", "")
                            else:
                                text = str(step)
                            st.write(f"{i}. {text}")
    else:
        st.caption("Ask anything about this lecture's transcript...")

    prompt = st.chat_input("Ask a question about the lecture...")
    if prompt:
        submit_question(prompt)
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)
