from pydantic import BaseModel, Field
from typing import List, Optional


class QueryRequest(BaseModel):
    """
    Request schema for query endpoint
    """
    question: str = Field(..., description="User question")
    session_id: Optional[str] = Field(None, description="Session ID for conversation tracking")
    stream: bool = Field(False, description="Enable streaming response")


class SourceInfo(BaseModel):
    """
    Source citation information
    """
    document: str
    page: Optional[int] = None
    relevance_score: float
    excerpt: str


class QueryResponse(BaseModel):
    """
    Response schema for query endpoint
    """
    answer: str
    sources: List[SourceInfo]
    confidence: float
    conversation_id: Optional[str] = None


class IngestResponse(BaseModel):
    """
    Response schema for ingest endpoint
    """
    status: str
    documents_processed: int
    chunks_created: int
    message: str
