"""
LangGraph module — orchestrates the Corrective RAG pipeline visually as a StateGraph.
"""

from typing import Annotated, Literal
import operator
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, HumanMessage

from langgraph.graph import StateGraph, START, END
try:
    from langgraph.checkpoint.sqlite import SqliteSaver
    _USE_SQLITE = True
except ImportError:
    from langgraph.checkpoint.memory import MemorySaver
    _USE_SQLITE = False
import sqlite3
import os

from modules import vector_store, relevance_evaluator
from modules import query_reformulator, web_search, context_aggregator
from config import settings
from modules.reranker import rerank
from modules.logger import get_logger
import time

log = get_logger(__name__)

# --- STATE DEFINITION ---

class AgentState(TypedDict):
    """The State of the LangGraph execution."""
    messages: Annotated[list[BaseMessage], operator.add]  # Conversational history
    original_query: str      # The user's actual question
    current_query: str       # The query being searched (might be reformulated)
    retrieved_docs: list[dict]  # Documents from FAISS
    web_docs: list[dict]        # Documents from arXiv web search
    evaluation: dict         # Relevance evaluation results
    correction_attempts: int
    used_web_fallback: bool
    final_answer: str
    confidence_data: dict
    aggregated_context: dict   # Stores the final aggregated context payload
    source_map: dict           # Stable label → title map (persisted across turns)
    steps_log: Annotated[list[dict], operator.add]  # Log of pipeline steps for the UI
    start_time: float


# --- NODE FUNCTIONS ---

def retrieve_node(state: AgentState) -> dict:
    """Retrieve documents from the vector store using the current query."""
    original_query = state["current_query"]
    chat_history = state.get("messages", [])[:-1] # Exclude the current human message

    # Contextualize the query using history (e.g. "explain it" -> "explain the GR-NLP paper")
    query = query_reformulator.contextualize(original_query, chat_history)
    
    step = {
        "step": f"retrieval (attempt {state['correction_attempts'] + 1})", 
        "status": "running", 
        "original_query": original_query,
        "contextualized_query": query if query != original_query else None
    }

    store = vector_store.get_vector_store()

    # FAISS similarity_search_with_score returns (Document, score) tuples
    try:
        results = store.similarity_search_with_score(query, k=settings.TOP_K)
    except Exception as e:
        log.warning(f"Retrieval error: {e}")
        results = []

    # Convert LangChain documents to the expected dict format for the evaluator.
    # FAISS returns L2 distance (lower = more similar). Convert to 0-1 similarity
    # so the relevance evaluator's threshold (e.g. 0.40) works correctly:
    #   distance 0.0 → similarity 1.0  (perfect match)
    #   distance 1.0 → similarity 0.5
    #   distance 9.0 → similarity 0.1  (far away / irrelevant)
    formatted_docs = []
    for doc, dist in results:
        similarity = round(1.0 / (1.0 + float(dist)), 4)  # cast numpy.float32 → Python float
        formatted_docs.append({
            "text": doc.page_content,
            "score": similarity,
            "metadata": doc.metadata
        })
    
    if len(formatted_docs) > 3:
        formatted_docs = rerank(query, formatted_docs, top_n=3)

    step["status"] = "done"
    step["results_count"] = len(formatted_docs)

    return {
        "retrieved_docs": formatted_docs,
        "steps_log": [step]
    }


def grade_documents_node(state: AgentState) -> dict:
    """Evaluate if the retrieved documents are relevant to the original query."""
    docs = state["retrieved_docs"]

    # Pass the original user question so keyword overlap can be computed
    evaluation = relevance_evaluator.evaluate(docs, query=state["original_query"])

    step = {
        "step": f"relevance_evaluation_{state['correction_attempts'] + 1}",
        "status": "done",
        "relevant": evaluation["relevant"],
        "avg_score": evaluation["avg_score"],
        "reason": evaluation.get("reason", ""),
    }

    return {
        "evaluation": evaluation,
        "steps_log": [step]
    }


def reformulate_query_node(state: AgentState) -> dict:
    """Rewrite the user query to hopefully get better retrieval results."""
    current_query = state["current_query"]
    attempts = state["correction_attempts"]

    step = {
        "step": f"query_reformulation_{attempts + 1}",
        "status": "running",
        "original_query": current_query,
    }

    # LRU-cached — identical queries don't fire extra API calls
    new_query = query_reformulator.reformulate(current_query)

    step["status"] = "done"
    step["reformulated_query"] = new_query

    return {
        "current_query": new_query,
        "correction_attempts": attempts + 1,
        "steps_log": [step]
    }


def web_search_node(state: AgentState) -> dict:
    """Fallback to searching arXiv directly if local retrieval fails too many times."""
    original_query = state["original_query"]

    step = {
        "step": "web_search_fallback",
        "status": "running",
        "query": original_query,
    }

    arxiv_results = web_search.search_arxiv(original_query, max_results=5)
    web_context_chunks = web_search.format_for_context(arxiv_results)

    step["status"] = "done"
    step["results_count"] = len(arxiv_results)

    return {
        "web_docs": web_context_chunks,
        "used_web_fallback": True,
        "steps_log": [step]
    }


def aggregate_node(state: AgentState) -> dict:
    """Final node: prepares the aggregated context for the UI/LLM."""
    local_docs = state.get("retrieved_docs", [])
    web_docs = state.get("web_docs", [])

    step = {"step": "context_aggregation", "status": "done"}
    aggregated = context_aggregator.aggregate(
        vector_results=local_docs,
        web_results=web_docs if web_docs else None,
    )

    # Merge with previous source_map so stable labels accumulate across turns
    prev_source_map = state.get("source_map") or {}
    new_source_map = {**prev_source_map, **aggregated.get("source_map", {})}

    return {
        "aggregated_context": aggregated,
        "source_map": new_source_map,
        "steps_log": [step]
    }


# --- CONDITIONAL EDGES ---

def check_relevance(state: AgentState) -> Literal["aggregate", "reformulate", "web_search"]:
    """Determine whether to generate, reformulate, or fallback to web search based on evaluation."""
    is_relevant = state["evaluation"]["relevant"]
    attempts = state["correction_attempts"]

    if is_relevant:
        return "aggregate"
    elif attempts < settings.MAX_CORRECTION_ATTEMPTS:
        return "reformulate"
    else:
        return "web_search"


# --- BUILD GRAPH ---

def build_graph() -> StateGraph:
    """Constructs and compiles the LangGraph with a persistent SQLite checkpointer."""
    workflow = StateGraph(AgentState)

    # Add Nodes
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade", grade_documents_node)
    workflow.add_node("reformulate", reformulate_query_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("aggregate", aggregate_node)

    # Build Edges
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade")

    workflow.add_conditional_edges(
        "grade",
        check_relevance,
        {
            "aggregate": "aggregate",
            "reformulate": "reformulate",
            "web_search": "web_search"
        }
    )

    workflow.add_edge("reformulate", "retrieve")
    workflow.add_edge("web_search", "aggregate")
    workflow.add_edge("aggregate", END)

    # Compile with persistent SQLite checkpointer (falls back to MemorySaver if
    # langgraph-checkpoint-sqlite is not installed)
    if _USE_SQLITE:
        os.makedirs("data", exist_ok=True)
        conn = sqlite3.connect("data/checkpoints.db", check_same_thread=False)
        memory = SqliteSaver(conn)
    else:
        memory = MemorySaver()

    app = workflow.compile(checkpointer=memory)
    return app


# Singleton App Instance (built only ONCE — bug fix: was built twice before)
carag_app = build_graph()
