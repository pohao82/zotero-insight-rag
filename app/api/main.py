from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn

# Importing your existing logic
from app.core.config import create_research_engine, retriever_module

app = FastAPI(title="Zotero RAG API Service")

# Initialize engines once at startup
# Note: create_research_engine now returns (langgraph_app, generator)
graph_engine, _ = create_research_engine()
retriever = retriever_module()

class ResearchRequest(BaseModel):
    question: str
    top_k: int = 5
    window: int = 2
    max_retries: int = 0
    # mode: "Semantic Search Only" or "Research Assistant (LLM)"
    search_mode: str = "Research Assistant (LLM)"
    mode: str = "standard"
    metadata_filters: Optional[Dict[str, Any]] = None

@app.get("/metadata/titles")
async def get_titles():
    try:
        # Use the already initialized retriever's database connection
        con = retriever.db.get_connection()
        rows = con.execute("SELECT DISTINCT title FROM paper_chunks").fetchall()
        return [row[0] for row in rows]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/research")
async def run_research(req: ResearchRequest):
    try:
        # Retrieval Phase
        context_text, sources, context_map = retriever.get_relevant_context(
            req.question, 
            top_k=req.top_k, 
            window=req.window, 
            filter_dict=req.metadata_filters, 
            mode=req.mode
        )

        # CONDITIONAL ROUTING return if semantic only
        if req.search_mode == "Semantic Search Only":
            # Return early - No LLM used
            return {
                "answer": "Here are the most relevant sections found in your library:",
                "verified": True,
                "feedback": "",
                "sources": sources,
                "context_map": context_map,
                "mode": "semantic"
            }

        # Agentic Execution (LangGraph)
        initial_input = {
            "question": req.question,
            "context": context_text,
            "iterations": 0,
            "verified": False,
            "feedback": "",
            "max_retries": req.max_retries
        }

        # Invoke the compiled LangGraph workflow
        final_state = graph_engine.invoke(initial_input)

        return {
            "answer": final_state["draft"],
            "verified": final_state["verified"],
            "feedback": final_state["feedback"],
            "sources": sources,
            "context_map": context_map
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
