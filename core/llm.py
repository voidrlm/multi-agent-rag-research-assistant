from langchain_openai import ChatOpenAI

from config.settings import get_settings


def get_llm(streaming: bool = True, temperature: float = 0.3) -> ChatOpenAI:
    settings = get_settings()
    if not settings.OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY is not set. Please add it to your .env file.\n"
            "Copy .env.example to .env and fill in your API key."
        )
    return ChatOpenAI(
        model=settings.OPENAI_MODEL,
        api_key=settings.OPENAI_API_KEY,
        streaming=streaming,
        temperature=temperature,
        max_tokens=4096,
    )
