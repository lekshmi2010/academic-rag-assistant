from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import config

# Define strict output schema using Pydantic
class LectureValidation(BaseModel):
    is_educational: bool = Field(
        description="True if the transcript snippet belongs to an educational lecture, computer science tutorial, academic presentation, or structured seminar. False if it is a casual vlog, gaming stream, talk show, or general entertainment."
    )
    confidence_score: float = Field(
        description="Confidence score for this classification between 0.0 and 1.0."
    )
    reasoning: str = Field(
        description="A clear, brief one-sentence explanation justifying the decision based on structural or lexical indicators in the text."
    )

def classify_transcript(text: str) -> bool:
    """
    Uses Gemini with structured output schema to classify if 
    a video transcript preview is educational or tutorial content.
    """
    if not config.GOOGLE_API_KEY or config.GOOGLE_API_KEY == "your_gemini_api_key_here":
        print("[Classifier Warning] GOOGLE_API_KEY is not configured. Falling back to False.")
        return False

    if not text or len(text.strip()) < 50:
        print("[Classifier Warning] Input text is too short for reliable classification.")
        return False

    try:
        # Initialize Gemini Flash (low latency, deterministic settings)
        llm = ChatGoogleGenerativeAI(
            model=config.CHAT_MODEL,
            google_api_key=config.GOOGLE_API_KEY,
            temperature=0.0
        )
        
        # Bind the schema to force JSON output matching our Pydantic model
        structured_llm = llm.with_structured_output(LectureValidation)
        
        # Build prompt focusing on structural cues
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are an expert academic routing agent. Your single task is to analyze video transcripts "
                "and determine if they represent structured educational content (like programming lectures, AI tutorials, "
                "or mathematical derivations) vs. entertainment media. Look for structural instruction indicators like "
                "defining terms, introducing syntax, sequential explanations, or conceptual reviews."
            ),
            ("user", "Transcript Preview:\n\n{text}")
        ])
        
        # Orchestrate the LangChain expression
        classifier_chain = prompt | structured_llm
        
        # Run the chain
        result = classifier_chain.invoke({"text": text})
        
        print(f"[Gemini Router] Approved: {result.is_educational} | Confidence: {result.confidence_score:.2f}")
        print(f"[Gemini Router] Reason: {result.reasoning}")
        
        # Enforce both the flag and a safety threshold
        return result.is_educational and result.confidence_score >= 0.50

    except Exception as e:
        print(f"[Gemini Router Error] Chain execution failed: {e}")
        return False

def classify_video_content(title: str, transcript: str, categories: list[str]) -> bool:
    """
    Uses Gemini to decide whether a video is educational based on its title,
    transcript, and YouTube categories.
    """
    if not config.GOOGLE_API_KEY or config.GOOGLE_API_KEY == "your_gemini_api_key_here":
        print("[Classifier Warning] GOOGLE_API_KEY is not configured. Falling back to False.")
        return False

    if not (title.strip() or transcript.strip() or categories):
        print("[Classifier Warning] Insufficient metadata for classification.")
        return False

    try:
        llm = ChatGoogleGenerativeAI(
            model=config.CHAT_MODEL,
            google_api_key=config.GOOGLE_API_KEY,
            temperature=0.0,
        )

        structured_llm = llm.with_structured_output(LectureValidation)

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are an expert academic routing agent. Decide whether a YouTube video is educational, "
                "academic, or tutorial content. Use the title, transcript, and categories together. Give most "
                "weight to the transcript, but use the title and categories as supporting evidence. Classify as "
                "educational when the content is structured instruction, explanation, derivation, lecture, or "
                "technical teaching. Otherwise classify it as not educational."
            ),
            (
                "user",
                "Title: {title}\n"
                "Categories: {categories}\n"
                "Transcript:\n\n{text}"
            ),
        ])

        classifier_chain = prompt | structured_llm
        result = classifier_chain.invoke(
            {
                "title": title.strip(),
                "categories": ", ".join(categories) if categories else "",
                "text": transcript.strip(),
            }
        )

        print(f"[Gemini Router] Approved: {result.is_educational} | Confidence: {result.confidence_score:.2f}")
        print(f"[Gemini Router] Reason: {result.reasoning}")

        return result.is_educational and result.confidence_score >= 0.50

    except Exception as e:
        print(f"[Gemini Router Error] Chain execution failed: {e}")
        return False
