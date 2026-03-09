import os
import streamlit as st
import uuid
import time
from langchain_core.messages import HumanMessage, AIMessage
from modules.graph import carag_app
from modules import vector_store
import logging

logging.getLogger("httpx").setLevel(logging.WARNING)

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="CRAG - Research Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    /* Dark Theme & Background */
    .stApp {
        background-color: #0d0d17;
        color: #e0e0e0;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #161625;
        border-right: 1px solid #2d2d3f;
    }
    
    /* Branding */
    .brand-container {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 30px;
    }
    .brand-logo {
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
        width: 40px;
        height: 40px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
    }
    .brand-name {
        font-size: 24px;
        font-weight: 700;
        letter-spacing: 1px;
        color: #ffffff;
    }
    .brand-sub {
        font-size: 10px;
        color: #7c7c9d;
        margin-top: -5px;
        margin-bottom: 20px;
    }
    
    /* Sidebar Headers */
    .sidebar-header {
        font-size: 11px;
        font-weight: 600;
        color: #7c7c9d;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin: 25px 0 15px 0;
    }
    
    /* Status Items */
    .status-item {
        display: flex;
        justify-content: space-between;
        margin-bottom: 12px;
        font-size: 14px;
    }
    .status-label { color: #94a3b8; }
    .status-value { color: #ffffff; font-weight: 600; }
    
    /* Process Steps */
    .process-step {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 15px;
    }
    .step-num {
        background: #1e1e30;
        border: 1px solid #2d2d3f;
        width: 24px;
        height: 24px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 11px;
        color: #818cf8;
    }
    .step-text { font-size: 13px; color: #cbd5e1; }
    
    /* Chat Header */
    .chat-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 30px;
        padding-bottom: 15px;
        border-bottom: 1px solid #2d2d3f;
    }
    .header-title { font-size: 20px; font-weight: 600; color: #ffffff; }
    .status-connected {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 12px;
        color: #10b981;
    }
    /* Pulse Animation */
    @keyframes pulse-green {
        0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(16, 185, 129, 0); }
        100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }
    .dot {
        width: 8px;
        height: 8px;
        background-color: #10b981;
        border-radius: 50%;
        animation: pulse-green 2s infinite;
    }

    /* Confidence Bar */
    .confidence-container {
        background: #161625;
        border: 1px solid #2d2d3f;
        border-radius: 12px;
        padding: 20px;
        margin-top: 20px;
    }
    .conf-header {
        display: flex;
        justify-content: space-between;
        margin-bottom: 12px;
    }
    .conf-title {
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #94a3b8;
    }
    .conf-value {
        font-size: 18px;
        font-weight: 700;
        color: #fbbf24;
    }
    .progress-bg {
        background: #1e1e30;
        height: 6px;
        border-radius: 3px;
        width: 100%;
        overflow: hidden;
    }
    .progress-fill {
        background: linear-gradient(90deg, #fbbf24 0%, #f59e0b 100%);
        height: 100%;
        transition: width 0.5s ease;
    }
    
    .metrics-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 20px;
        margin-top: 20px;
        font-size: 13px;
    }
    .metric-row { display: flex; justify-content: space-between; color: #7c7c9d; }
    .metric-v { color: #cbd5e1; }

    /* Buttons */
    .stButton>button {
        width: 100%;
        background-color: #1e1e30 !important;
        border: 1px solid #2d2d3f !important;
        color: #ffffff !important;
        border-radius: 8px !important;
        padding: 10px !important;
        font-size: 14px !important;
        transition: all 0.2s !important;
    }
    .stButton>button:hover {
        border-color: #6366f1 !important;
        background-color: #272744 !important;
    }
</style>
""", unsafe_allow_html=True)

# --- INITIALIZE STATE ---
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "doc_count" not in st.session_state:
    try:
        st.session_state.doc_count = vector_store.get_collection_count()
    except:
        st.session_state.doc_count = 0

# --- SIDEBAR ---
with st.sidebar:
    # Branding
    st.markdown("""
    <div class="brand-container">
        <div class="brand-logo">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2004/svg">
                <path d="M12 2L2 7L12 12L22 7L12 2Z" fill="white"/>
                <path d="M2 17L12 22L22 17" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                <path d="M2 12L12 17L22 12" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
        </div>
        <div class="brand-name">CRAG</div>
    </div>
    <div class="brand-sub">Corrective Agentic RAG</div>
    """, unsafe_allow_html=True)

    # System Status
    st.markdown('<div class="sidebar-header">System Status</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="status-item">
        <span class="status-label">Vector Store</span>
        <span class="status-value">{st.session_state.doc_count} docs</span>
    </div>
    <div class="status-item">
        <span class="status-label">Model</span>
        <span class="status-value">Gemini 2.0</span>
    </div>
    <div class="status-item">
        <span class="status-label">Embeddings</span>
        <span class="status-value">MiniLM-L6</span>
    </div>
    """, unsafe_allow_html=True)

    # Actions
    st.markdown('<div class="sidebar-header">Actions</div>', unsafe_allow_html=True)
    if st.button("📥 Ingest Papers"):
        with st.status("Ingesting papers..."):
            from ingest import ingest
            ingest()
            st.session_state.doc_count = vector_store.get_collection_count()
        st.rerun()
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

    # How it Works
    st.markdown('<div class="sidebar-header">How it Works</div>', unsafe_allow_html=True)
    steps_list = [
        "Vector Retrieval",
        "Relevance Check",
        "Query Correction",
        "Web Fallback",
        "Answer + Score"
    ]
    for i, step in enumerate(steps_list, 1):
        st.markdown(f"""
        <div class="process-step">
            <div class="step-num">{i}</div>
            <div class="step-text">{step}</div>
        </div>
        """, unsafe_allow_html=True)

# --- MAIN HEADER ---
st.markdown("""
<div class="chat-header">
    <div class="header-title">Research Assistant</div>
    <div class="status-connected">
        <div class="dot"></div>
        Connected
    </div>
</div>
""", unsafe_allow_html=True)

# --- MESSAGES ---
# Display history
for message in st.session_state.messages:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

# --- INPUT & ORCHESTRATION ---
if prompt := st.chat_input("Ask about AI/ML research..."):
    # Display user input
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append(HumanMessage(content=prompt))

    # Run LangGraph
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    input_state = {
        "messages": [HumanMessage(content=prompt)],
        "original_query": prompt,
        "current_query": prompt,
        "correction_attempts": 0,
        "used_web_fallback": False,
        "web_docs": [],
        "steps_log": [],
        "start_time": time.time()
    }
    # Carry the stable source_map forward from the previous turn
    prev_state = carag_app.get_state(config).values
    if prev_state.get("source_map"):
        input_state["source_map"] = prev_state["source_map"]

    with st.chat_message("assistant"):
        status_info = st.empty()
        detailed_expander = st.expander("Pipeline Steps", expanded=False)
        
        try:
            # 1. Run the graph logic (Retrieval, Grading, Reformulation)
            # Graph now stops at Context Aggregation
            final_graph_state = None
            for event in carag_app.stream(input_state, config):
                for node_name, node_output in event.items():
                    status_info.markdown(f"*(Agent Node: **{node_name}**)*")
                    final_graph_state = node_output
                    
                    steps = node_output.get("steps_log", [])
                    for s in steps:
                        with detailed_expander:
                            if "retrieval" in s['step']: icon = "⚡"
                            elif "relevance" in s['step']: icon = "🟢" if s.get('relevant') else "🔴"
                            elif "reformulation" in s['step']: icon = "🔄"
                            elif "web" in s['step']: icon = "🌍"
                            else: icon = "🔍"
                            st.write(f"{icon} **{s['step']}** completed")

            # 2. Get the Final State for generation
            # We access the compiled graph's state to get everything
            from modules import llm_generator, confidence_scorer
            state_values = carag_app.get_state(config).values
            agg = state_values.get("aggregated_context", {})
            context_text = agg.get("context_text", "")
            
            # 3. Stream the final answer
            status_info.markdown("*(Generating final answer...)*")
            
            # Full history (all prior turns) for conversation memory
            chat_history = st.session_state.messages[:-1]
            # Stable source map from this turn's graph state
            source_map = state_values.get("source_map", {})
            
            answer_placeholder = st.empty()
            full_response = ""
            
            for chunk in llm_generator.stream_generate(prompt, context_text, chat_history, source_map):
                full_response += chunk
                display_text = full_response.split("SELF_SCORE:")[0].strip()
                answer_placeholder.markdown(f"### CRAG Assistant\n\n{display_text}")
            
            # Extract final answer and score
            final_answer, self_score = llm_generator.parse_score(full_response)
            status_info.empty()
            answer_placeholder.markdown(f"### CRAG Assistant\n\n{final_answer}")
            
            # 4. Compute Confidence & Scorecard
            eval_data = state_values.get("evaluation", {"avg_score": 0.0, "scores": []})
            conf_result = confidence_scorer.compute(
                avg_similarity=eval_data["avg_score"],
                retrieval_scores=eval_data["scores"],
                llm_self_score=self_score,
                correction_attempts=state_values.get("correction_attempts", 0),
                used_web_fallback=state_values.get("used_web_fallback", False)
            )
            
            score_pct = conf_result["confidence"] * 100
            breakdown = conf_result["breakdown"]
            
            st.markdown(f"""
            <div class="confidence-container">
                <div class="conf-header">
                    <div class="conf-title">Confidence Score</div>
                    <div class="conf-value">{score_pct:.1f}%</div>
                </div>
                <div class="progress-bg">
                    <div class="progress-fill" style="width: {score_pct}%"></div>
                </div>
                <div class="metrics-grid">
                    <div class="metric-row">
                        <span>Similarity:</span>
                        <span class="metric-v">{breakdown.get('similarity', 0)*100:.0f}%</span>
                    </div>
                    <div class="metric-row">
                        <span>Consistency:</span>
                        <span class="metric-v">{breakdown.get('consistency', 0)*100:.0f}%</span>
                    </div>
                    <div class="metric-row">
                        <span>LLM Score:</span>
                        <span class="metric-v">{breakdown.get('llm_self_score', 0)*100:.0f}%</span>
                    </div>
                    <div class="metric-row">
                        <span>Corrections:</span>
                        <span class="metric-v">{breakdown.get('correction_attempts', 0)}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Summary metrics bar
            st.markdown(f"""
            <div style="display: flex; gap: 20px; font-size: 11px; color: #7c7c9d; margin-top: 15px; opacity: 0.8;">
                <span>⚡ {state_values.get('correction_attempts', 0)} corrections</span>
                <span>📚 {len(agg.get('sources', []))} Sources</span>
                <span>🌍 {'Web Fallback' if state_values.get('used_web_fallback') else 'Local KB Only'}</span>
            </div>
            """, unsafe_allow_html=True)
            
            # Sources Tab
            sources = agg.get("sources", [])
            if sources:
                with st.expander(f"📚 {len(sources)} Sources", expanded=False):
                    for s in sources:
                        st.markdown(f"**[{s['id']}] {s.get('title', 'Untitled')}**")
                        st.markdown(f"*Authors: {s.get('authors', 'Unknown')}*")
                        if s.get('url'): st.markdown(f"[Source Link]({s['url']})")
                        st.text(s.get('preview', '')[:500] + "...")
                        st.divider()

            # 5. Save and finalize state
            st.session_state.messages.append(AIMessage(content=final_answer))
            
            # Persist the final answer, confidence data, AND the stable source_map
            carag_app.update_state(config, {
                "messages": [AIMessage(content=final_answer)],
                "final_answer": final_answer,
                "confidence_data": conf_result,
                "source_map": state_values.get("source_map", {})
            })

        except Exception as e:
            st.error(f"Execution Error: {str(e)}")
            st.exception(e)
