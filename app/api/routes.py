from fastapi import APIRouter, HTTPException
import logging
from .schemas import QueryRequest, QueryResponse, IngestResponse, SourceInfo
from ..graph.workflow import run_rag_query
from ..ingestion.loader import DocumentLoader, load_sample_documents
from ..retrieval.vector_store import add_documents

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query the RAG system with a question
    """
    try:
        logger.info(f"Received query: {request.question}")

        # Run RAG workflow
        result = run_rag_query(
            question=request.question,
            session_id=request.session_id
        )

        # Format response
        sources = [
            SourceInfo(**source)
            for source in result.get("sources", [])
        ]

        response = QueryResponse(
            answer=result.get("answer", ""),
            sources=sources,
            confidence=result.get("confidence", 0.0),
            conversation_id=request.session_id
        )

        return response

    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingest", response_model=IngestResponse)
async def ingest_documents():
    """
    Ingest documents from the sample-docs directory into vector store
    """
    try:
        logger.info("Starting document ingestion")

        # Try to load documents from directory
        loader = DocumentLoader()
        chunked_docs = loader.load_and_chunk()

        # If no documents found, use sample documents
        if not chunked_docs:
            logger.info("No documents found in directory, using sample documents")
            chunked_docs = load_sample_documents()

        # Add to vector store
        add_documents(chunked_docs)

        response = IngestResponse(
            status="success",
            documents_processed=len(chunked_docs),
            chunks_created=len(chunked_docs),
            message=f"Successfully ingested {len(chunked_docs)} document chunks"
        )

        logger.info(f"Ingestion completed: {response.message}")

        return response

    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_stats():
    """
    Get statistics about the vector store
    """
    try:
        from ..retrieval.vector_store import get_vector_store

        vector_store = get_vector_store()

        # Get collection stats
        collection = vector_store._collection
        count = collection.count()

        return {
            "total_documents": count,
            "collection_name": collection.name,
            "status": "active"
        }

    except Exception as e:
        logger.error(f"Stats retrieval failed: {e}")
        return {
            "total_documents": 0,
            "status": "not_initialized",
            "error": str(e)
        }
