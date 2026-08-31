def normalize_answer_content(content) -> str:
    """
    Normalizes LangChain/Gemini text content blocks (including lists and dicts)
    into a plain continuous string.
    """
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if text:
                    text_parts.append(str(text))
        if text_parts:
            return "\n".join(text_parts)

    return str(content or "")
