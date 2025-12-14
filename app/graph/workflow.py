from langgraph.graph import StateGraph, END
from .state import GraphState
from .nodes import (
    query_analysis_node,
    retrieval_node,
    relevance_check_node,
    generation_node,
    source_attribution_node,
    fallback_node,
    clarification_node
)
import logging

logger = logging.getLogger(__name__)


def should_retrieve(state: GraphState) -> str:
    """
    Conditional edge: determine if retrieval is needed
    """
    if state.get("needs_clarification"):
        return "clarification"
    elif state.get("needs_retrieval"):
        return "retrieval"
    else:
        return "generation"


def should_generate(state: GraphState) -> str:
    """
    Conditional edge: determine if we can generate or need fallback
    """
    confidence = state.get("confidence", 0.0)
    threshold = 0.3

    if confidence >= threshold:
        return "generation"
    else:
        return "fallback"


def create_workflow() -> StateGraph:
    """
    Create the LangGraph workflow for RAG
    """
    # Initialize graph
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("query_analysis", query_analysis_node)
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("relevance_check", relevance_check_node)
    workflow.add_node("generation", generation_node)
    workflow.add_node("source_attribution", source_attribution_node)
    workflow.add_node("fallback", fallback_node)
    workflow.add_node("clarification", clarification_node)

    # Set entry point
    workflow.set_entry_point("query_analysis")

    # Add edges
    # Query analysis -> retrieval or clarification
    workflow.add_conditional_edges(
        "query_analysis",
        should_retrieve,
        {
            "retrieval": "retrieval",
            "clarification": "clarification",
            "generation": "generation"
        }
    )

    # Retrieval -> relevance check
    workflow.add_edge("retrieval", "relevance_check")

    # Relevance check -> generation or fallback
    workflow.add_conditional_edges(
        "relevance_check",
        should_generate,
        {
            "generation": "generation",
            "fallback": "fallback"
        }
    )

    # Generation -> source attribution -> END
    workflow.add_edge("generation", "source_attribution")
    workflow.add_edge("source_attribution", END)

    # Fallback -> END
    workflow.add_edge("fallback", END)

    # Clarification -> END
    workflow.add_edge("clarification", END)

    return workflow.compile()


# Create the compiled workflow
rag_workflow = create_workflow()


def run_rag_query(question: str, session_id: str = None) -> dict:
    """
    Run a RAG query through the workflow
    """
    logger.info(f"Running RAG query: {question}")

    # Initialize state
    initial_state = {
        "question": question,
        "session_id": session_id,
        "chat_history": [],
        "retrieved_documents": [],
        "retrieval_query": None,
        "needs_retrieval": False,
        "needs_clarification": False,
        "clarification_question": None,
        "answer": "",
        "confidence": 0.0,
        "sources": [],
        "steps_taken": [],
        "error": None
    }

    # Run workflow
    try:
        result = rag_workflow.invoke(initial_state)
        logger.info(f"Workflow completed. Steps: {result.get('steps_taken')}")
        return result
    except Exception as e:
        logger.error(f"Workflow execution failed: {e}")
        return {
            "answer": "An error occurred while processing your question.",
            "error": str(e),
            "confidence": 0.0,
            "sources": []
        }
