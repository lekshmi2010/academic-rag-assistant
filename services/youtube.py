import re
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from fastapi import HTTPException

def _fetch_transcript_snippets(video_id: str):
    """
    Fetches transcript snippets for the requested video using the instance-based 
    youtube-transcript-api flow.
    """
    transcript_api = YouTubeTranscriptApi()
    transcript_list = transcript_api.list(video_id)
    transcript = transcript_list.find_transcript(["en"])
    return transcript.fetch()

def _snippet_text(snippet) -> str:
    return snippet.text if hasattr(snippet, "text") else snippet["text"]

def _duration_to_seconds(info: dict) -> float:
    duration = info.get("duration")
    if duration:
        return float(duration)

    duration_text = info.get("duration_string")
    if not duration_text:
        return 0

    parts = duration_text.split(":")
    if not all(part.isdigit() for part in parts):
        return 0

    total = 0
    for part in parts:
        total = total * 60 + int(part)

    return float(total)

def extract_video_id(url: str) -> str:
    """
    Extracts the unique 11-character video ID from various YouTube URL formats.
    """
    pattern = r'(?:https?://)?(?:www\.)?(?:youtube\.com/(?:[^/]+/.+/|(?:v|e(?:mbed)?)/||.*[?&]v=)|youtu\.be/)([^"&?/\s]{11})'
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    raise ValueError("Could not extract a valid YouTube Video ID from the provided URL.")

def fetch_full_transcript(video_id: str) -> str:
    """
    Fetches the complete transcript for a given YouTube video ID 
    and concatenates all text fragments into a single continuous string.
    """
    try:
        transcript_list = _fetch_transcript_snippets(video_id)
        full_text = " ".join([_snippet_text(segment) for segment in transcript_list])
        cleaned_text = " ".join(full_text.split())
        return cleaned_text
    except TranscriptsDisabled:
        raise HTTPException(status_code=400, detail="Captions/Transcripts are disabled for this video.")
    except NoTranscriptFound:
        raise HTTPException(status_code=400, detail="No English or auto-generated transcripts found.")
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to fetch full transcript for video {video_id}: {str(e)}"
        )

def fetch_video_metadata(video_url: str) -> dict:
    """
    Extracts deep metadata fields from a YouTube video to evaluate its
    structural layout and academic signaling.
    """
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'extract_flat': False,
        'noplaylist': True,
        'skip_download': True,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)
            
            chapters = info.get('chapters') or []
            categories = info.get('categories') or []
            description = info.get('description') or ""
            duration = _duration_to_seconds(info)
            if duration <= 0:
                raise ValueError("Could not determine video duration from YouTube metadata.")
            
            # Scrape description for technical/academic anchors
            has_github = "github.com" in description.lower()
            has_arxiv = "arxiv.org" in description.lower() or "pdf" in description.lower()
            has_colab = "colab.research.google.com" in description.lower()
            
            # Count outlinks found in text
            url_count = len(re.findall(r'https?://[^\s]+', description))

            return {
                "duration_minutes": duration / 60,
                "categories": categories,
                "chapter_count": len(chapters),
                "has_technical_links": has_github or has_arxiv or has_colab,
                "total_link_count": url_count,
                "title": info.get('title', '')
            }
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"[Metadata Error] Failed extraction: {e}")
        raise HTTPException(status_code=502, detail=f"Failed to fetch YouTube metadata: {str(e)}")
