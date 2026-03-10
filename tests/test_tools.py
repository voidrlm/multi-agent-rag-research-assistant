import os


def test_search_tool_fallback_to_duckduckgo():
    # Ensure no Tavily key
    old = os.environ.pop("TAVILY_API_KEY", None)
    try:
        from config.settings import get_settings
        get_settings.cache_clear()

        from tools.search import get_search_tool
        tool = get_search_tool()
        assert "duckduckgo" in tool.name.lower() or "duck" in type(tool).__name__.lower()
    finally:
        if old:
            os.environ["TAVILY_API_KEY"] = old
        get_settings.cache_clear()
