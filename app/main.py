from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import logging
from datetime import datetime

from .api.routes import router as api_router
from .api.schemas import QueryRequest, QueryResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="LangGraph RAG Assistant API",
    description="Production-ready RAG system with LangGraph workflows",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router)


@app.get("/")
async def root():
    """
    API root endpoint
    """
    return {
        "message": "LangGraph RAG Assistant API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint for container orchestration
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "rag-assistant"
    }


@app.on_event("startup")
async def startup_event():
    """
    Run on application startup
    """
    logger.info("Starting LangGraph RAG Assistant API")
    logger.info("Initializing vector store...")

    # Initialize vector store and load documents
    try:
        from .retrieval.vector_store import get_vector_store
        vector_store = get_vector_store()
        logger.info("Vector store initialized successfully")
    except Exception as e:
        logger.warning(f"Vector store initialization failed: {e}")
        logger.warning("Documents may need to be ingested via /ingest endpoint")


@app.on_event("shutdown")
async def shutdown_event():
    """
    Run on application shutdown
    """
    logger.info("Shutting down LangGraph RAG Assistant API")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
