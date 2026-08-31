from fastapi import HTTPException
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI

import config
from database import get_vector_store
from utils.text import normalize_answer_content

SESSION_CHAT_HISTORY: dict[str, list] = {}

def _format_context(documents) -> str:
    """Join retrieved documents into the context block consumed by the prompt."""
    return "\n\n".join(document.page_content for document in documents if document.page_content)

def _get_session_history(session_id: str) -> list:
    return SESSION_CHAT_HISTORY.setdefault(session_id, [])

def _trim_history(history: list) -> list:
    return history[-config.MAX_HISTORY_MESSAGES:]

def ask_lecture_question(video_id: str, user_question: str, session_id: str) -> str:
    """
    Search the local vector database for transcript chunks that belong to the
    requested video and answer the user's question from that context.
    """
    if not config.GOOGLE_API_KEY or config.GOOGLE_API_KEY == "your_gemini_api_key_here":
        raise HTTPException(status_code=500, detail="Google API Key is missing.")

    if not video_id:
        raise HTTPException(status_code=400, detail="video_id is required.")

    if not user_question or not user_question.strip():
        raise HTTPException(status_code=400, detail="user_question is required.")

    if not session_id or not session_id.strip():
        raise HTTPException(status_code=400, detail="session_id is required.")

    try:
        vector_store = get_vector_store()
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 5,
                "filter": {"video_id": video_id},
            },
        )

        documents = retriever.invoke(user_question)
        context = _format_context(documents)
        history = _trim_history(_get_session_history(session_id.strip()))

        llm = ChatGoogleGenerativeAI(
            model=config.CHAT_MODEL,
            google_api_key=config.GOOGLE_API_KEY,
            temperature=0.3,
        )

        system_prompt = (
            "You are an expert AI teaching assistant helping students learn from video lectures. "
            "Your answers must be grounded exclusively in the transcript context provided — do not "
            "use outside knowledge or fabricate details. If the answer is not in the context, "
            "clearly say: 'This isn't covered in the video.'\n\n"

            "## Transcript Noise Handling\n"
            "The context comes from auto-generated YouTube captions and may contain:\n"
            "- Phonetic errors: 'we go of n' → O(n), 'wile loop' → while loop, 'nueral' → neural\n"
            "- Missing punctuation, run-on sentences, filler words (uh, um, you know)\n"
            "- Garbled math: 'x squared plus 2x' → x² + 2x\n"
            "Silently correct these in your output — never quote the raw typo.\n\n"

            "## Response Guidelines\n"
            "- **Explain, don't just retrieve.** Rephrase concepts in clear academic language; "
            "don't copy transcript sentences verbatim.\n"
            "- **Match the student's level.** If they ask a basic question, keep it simple. "
            "If they ask a follow-up with technical depth, go deeper.\n"
            "- **Use structure when helpful.** For multi-step concepts, use numbered steps or "
            "bullet points. For definitions, lead with a crisp one-liner then elaborate.\n"
            "- **Cite the video implicitly.** Use phrases like 'As explained in the lecture...' "
            "or 'The instructor describes this as...' to reinforce that answers come from the video.\n"
            "- **Encourage active learning.** When appropriate, end with a clarifying question "
            "or suggest what to review next (e.g., 'You may also want to revisit the section on X').\n\n"

            "## Boundaries\n"
            "- Never answer questions unrelated to the video content.\n"
            "- If context is ambiguous or incomplete, say so honestly rather than guessing.\n"
            "- Do not provide code solutions or homework answers directly — guide instead.\n\n"

            "Context (from video transcript):\n{context}"
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}"),
            ]
        )

        chain = prompt | llm
        response = chain.invoke({"context": context, "input": user_question, "history": history})
        answer_text = normalize_answer_content(response.content if hasattr(response, "content") else response)

        session_history = _get_session_history(session_id.strip())
        session_history.append(HumanMessage(content=user_question))
        session_history.append(AIMessage(content=answer_text))
        SESSION_CHAT_HISTORY[session_id.strip()] = _trim_history(session_history)

        return answer_text

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat generation failed: {str(e)}")
