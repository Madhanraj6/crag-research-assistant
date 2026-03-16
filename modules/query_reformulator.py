"""
Query Reformulator — uses LangChain ChatOpenAI to rewrite queries for better retrieval.
Results are cached so the same query never fires more than one LLM call.
"""

import functools
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from config import settings
from modules.logger import get_logger

log = get_logger(__name__)

_llm = None

REFORMULATION_TEMPLATE = """You are a search query optimizer. Your task is to reformulate the user's query to improve retrieval from a vector database of AI/ML research papers.

Original query: {query}

Previous retrieval was insufficient. Rewrite the query to be more specific and likely to match relevant research paper abstracts. Focus on:
1. Using precise technical terminology
2. Including related concepts and synonyms
3. Being specific about the research area

Return ONLY the reformulated query text, nothing else."""

CONTEXTUALIZE_TEMPLATE = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is.

Chat History:
{history_str}

Latest Question: {query}

Return ONLY the standalone question text, nothing else."""


def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model="google/gemini-2.0-flash-001",
            base_url="https://openrouter.ai/api/v1",
            api_key=settings.OPENROUTER_API_KEY,
            temperature=0.2,
        )
    return _llm


@functools.lru_cache(maxsize=128)
def reformulate(query: str) -> str:
    """
    Use LangChain to rewrite a query for better vector retrieval.
    Results are cached — identical queries never make more than one API call.
    """
    llm = _get_llm()
    prompt = PromptTemplate.from_template(REFORMULATION_TEMPLATE)
    chain = prompt | llm

    try:
        response = chain.invoke({"query": query})
        reformulated = response.content.strip()
        if reformulated.startswith('"') and reformulated.endswith('"'):
            reformulated = reformulated[1:-1]
        return reformulated
    except Exception as e:
        log.warning(f"Reformulation error: {e}")
        return query


def contextualize(query: str, chat_history: list) -> str:
    """
    Rewrite a conversational query (e.g. 'explain that paper') into a standalone
    query (e.g. 'explain the GR-NLP-TOOLKIT paper') using the chat history.
    """
    if not chat_history:
        return query

    # Format history into a readable string
    history_lines = []
    for msg in chat_history[-12:]:  # Look at last 6 turns
        role = "User" if msg.type == "human" else "Assistant"
        content = msg.content
        if len(content) > 300:
            content = content[:300] + "..."  # Truncate long assistant replies
        history_lines.append(f"{role}: {content}")
    history_str = "\n".join(history_lines)

    llm = _get_llm()
    prompt = PromptTemplate.from_template(CONTEXTUALIZE_TEMPLATE)
    chain = prompt | llm

    try:
        response = chain.invoke({"query": query, "history_str": history_str})
        standalone = response.content.strip()
        if standalone.startswith('"') and standalone.endswith('"'):
            standalone = standalone[1:-1]
        return standalone
    except Exception as e:
        log.warning(f"Contextualize error: {e}")
        return query
