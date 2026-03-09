# CRAG Research Assistant
**Corrective Agentic Retrieval-Augmented Generation (CRAG)**

An AI-powered research assistant built with **Streamlit** and **LangGraph** that retrieves and aggregates academic papers to answer complex questions. The app uses a local **FAISS** vector database for the knowledge base, with a corrective loop that dynamically queries the live **arXiv API** if the local documents fail relevance checks. It includes conversation memory, stable source citation, and confidence scoring.

## 🌟 Features
- **Agentic Corrections (LangGraph):** Built-in loop to grade retrieved docs. If they fail (score < threshold or missing keywords), it uses an LLM to reformulate the query or falls back to live web search.
- **Local FAISS Index:** Extremely fast local vector similarity search holding hundreds of arXiv papers.
- **Strict Source Grounding:** The LLM is forced by a strict prompt architecture to only attribute facts to the specific [KB: ...] or [WEB: ...] tags generated in the context.
- **Contextual Memory:** Intercepts follow-ups like "explain that paper" and rewrites them into standalone queries like "explain the GR-NLP paper" before retrieval, maintaining conversation continuity.
- **Streaming UI:** Instant word-by-word streaming generation from Gemini 2.0 Flash via OpenRouter.
- **Confidence Grader:** Computes a scorecard based on FAISS distance, relevance consistency, and the LLM's own self-reported confidence.

## 🏗️ Architecture

You can find full ASCII flowcharts of how each Python module works in the `docs/` folder.

1. **Contextualize (Query Rewrite):** Resolves vague follow-up questions using chat history.
2. **Retrieve:** Queries local FAISS vector store.
3. **Grade:** A two-layer check (Similarity score + Keyword overlap) acts as a strict relevance bouncer.
4. **Reformulate / Web Fallback:** If retrieval fails, the agent writes a better query or scrapes live arXiv.
5. **Aggregate:** Organizes sources into a clean, numbered context block with stable arXiv IDs.
6. **Stream Answer:** Prompts Gemini using the aggregated context and exact source citations.

## 🚀 Getting Started Locally

### Prerequisites
- Python 3.10+
- An API Key from OpenRouter

### Installation

1. **Clone the repo**
   ```bash
   git clone https://github.com/your-username/crag-research-assistant.git
   cd crag-research-assistant
   ```

2. **Run setup & ingest**
   Create a `.env` file based on `.env.example`:
   ```bash
   cp .env.example .env
   # Edit .env and add your OPENROUTER_API_KEY
   ```

   Install dependencies and build the FAISS index:
   ```bash
   pip install -r requirements.txt
   python ingest.py
   ```
   *(This downloads ~300 papers from arXiv cs.AI, cs.LG, cs.CL and populates `data/faiss_index`)*

3. **Run the App**
   ```bash
   streamlit run app.py
   ```

