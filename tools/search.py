import os

from langchain_core.tools import BaseTool

from config.settings import get_settings


def get_search_tool() -> BaseTool:
    settings = get_settings()

    if settings.TAVILY_API_KEY:
        # TavilySearchAPIWrapper reads from env var directly, so set it
        os.environ["TAVILY_API_KEY"] = settings.TAVILY_API_KEY
        from langchain_community.tools.tavily_search import TavilySearchResults
        return TavilySearchResults(max_results=5)

    from langchain_community.tools import DuckDuckGoSearchResults
    return DuckDuckGoSearchResults(max_results=5)
