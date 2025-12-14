from typing import TypedDict, List, Optional
from pydantic import BaseModel


class Document(BaseModel):
    """
    Document model for retrieved context
    """
    content: str
    metadata: dict
    relevance_score: Optional[float] = None


class GraphState(TypedDict):
    """
    State object for the LangGraph workflow
    """
    # User input
    question: str
    session_id: Optional[str]

    # Conversation history
    chat_history: List[dict]

    # Retrieval
    retrieved_documents: List[Document]
    retrieval_query: Optional[str]

    # Analysis
    needs_retrieval: bool
    needs_clarification: bool
    clarification_question: Optional[str]

    # Generation
    answer: str
    confidence: float

    # Sources
    sources: List[dict]

    # Metadata
    steps_taken: List[str]
    error: Optional[str]
