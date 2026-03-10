from functools import lru_cache

from langgraph.checkpoint.memory import MemorySaver


@lru_cache
def get_checkpointer() -> MemorySaver:
    return MemorySaver()
