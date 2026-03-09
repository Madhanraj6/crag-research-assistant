"""
LLM Answer Generator — uses LangChain ChatOpenAI to generate answers from provided context.
The system prompt enforces strict source grounding: the model must cite sources by their
exact label (e.g. [KB: arXiv:2304.12345]) and must not attribute statements to the wrong source.
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from config import settings

_llm = None

SYSTEM_PROMPT = """You are an expert AI research assistant.

## Primary Rule: Use Context + Cite
When the provided context contains relevant information, use it and cite every factual claim
with the EXACT source label from the context, e.g. [KB: arXiv:2304.12345].

## Conversation History Rule
You have access to the chat history. If the user asks a conversational question (e.g. "what did I ask before?"), answer it directly using the chat history without requiring context citations. Do not say you don't have enough info if the answer is in the chat history.

## Foundational Knowledge Rule
If the user asks about a general AI/ML concept (e.g., "what is deep learning", "explain transformers", "how does attention work") OR asks you to explain/simplify a previous answer (e.g., "explain in simple terms"):
- First, answer the question or explain the concept clearly using your general AI knowledge.
- Explicitly state: "Based on general AI knowledge:"
- Only then, reference how the retrieved papers build on or relate to that concept, if applicable.

## Citation Rules
- Do NOT invent citations or use a label for a different paper than what appears in context.
- If the user refers to a source label (e.g. "explain WEB-1"), check the SOURCE MAP first.
- A SOURCE MAP is injected into the prompt when available — always use it to resolve labels.

## Format
- Clear paragraphs, inline citations like [KB: arXiv:XXXX].
- If context is completely off-topic, not foundational AI, and not about chat history, say you don't have enough info.

After your answer, on a new line:
SELF_SCORE: 0.85"""


def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model="google/gemini-2.0-flash-001",
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=settings.OPENROUTER_API_KEY,
            temperature=0.2,
        )
    return _llm


def _build_human_prompt(query: str, context_text: str, source_map: dict | None) -> str:
    """Build the human prompt that injects context + source map."""
    map_block = ""
    if source_map:
        map_lines = "\n".join(f"  {label}  →  {title}" for label, title in source_map.items())
        map_block = f"\n\nSOURCE MAP (label → paper title):\n{map_lines}\n"

    context_block = context_text.strip() if context_text.strip() else "[NO CONTEXT — use only knowledge you are highly confident about and state that clearly]"

    return f"""Context:{map_block}

{context_block}

Question: {query}

Provide a grounded answer. If the question is about our conversation history, answer it directly using the chat history provided. Otherwise, use the context sources above and cite each claim with its exact label.
Remember to include SELF_SCORE at the end."""


def stream_generate(query: str, context_text: str, chat_history: list = None, source_map: dict = None):
    """Generator that streams the answer tokens."""
    llm = _get_llm()
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    if chat_history:
        messages.extend(chat_history)
    messages.append(HumanMessage(content=_build_human_prompt(query, context_text, source_map)))

    for chunk in llm.stream(messages):
        yield chunk.content


def parse_score(text: str) -> tuple[str, float]:
    """Extract answer and self-score from the full text."""
    answer = text
    self_score = 0.5
    if "SELF_SCORE:" in text:
        parts = text.rsplit("SELF_SCORE:", 1)
        answer = parts[0].strip()
        try:
            self_score = float(parts[1].strip())
            self_score = max(0.0, min(1.0, self_score))
        except ValueError:
            pass
    return answer, self_score


def generate(query: str, context_text: str, chat_history: list = None, source_map: dict = None) -> dict:
    """Non-streaming answer generation."""
    llm = _get_llm()
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    if chat_history:
        messages.extend(chat_history)
    messages.append(HumanMessage(content=_build_human_prompt(query, context_text, source_map)))

    try:
        response = llm.invoke(messages)
        raw_text = response.content.strip()
        answer, self_score = parse_score(raw_text)
        return {"answer": answer, "self_score": self_score, "raw_response": raw_text}
    except Exception as e:
        return {"answer": f"Error generating answer: {str(e)}", "self_score": 0.0, "raw_response": ""}
