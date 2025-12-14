import os
from typing import List
import logging
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document

logger = logging.getLogger(__name__)

# Global vector store instance
_vector_store = None

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
COLLECTION_NAME = "technical_docs"


def get_embeddings():
    """
    Get embeddings model
    """
    return OpenAIEmbeddings(
        model="text-embedding-ada-002"
    )


def get_vector_store() -> Chroma:
    """
    Get or create vector store instance
    """
    global _vector_store

    if _vector_store is None:
        logger.info(f"Initializing ChromaDB at {CHROMA_PERSIST_DIR}")

        embeddings = get_embeddings()

        # Create or load vector store
        _vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=CHROMA_PERSIST_DIR
        )

        logger.info("ChromaDB initialized successfully")

    return _vector_store


def add_documents(documents: List[Document]) -> None:
    """
    Add documents to vector store
    """
    vector_store = get_vector_store()

    logger.info(f"Adding {len(documents)} documents to vector store")

    vector_store.add_documents(documents)

    logger.info("Documents added successfully")


def search_documents(query: str, k: int = 5) -> List[Document]:
    """
    Search for documents similar to query
    """
    vector_store = get_vector_store()

    results = vector_store.similarity_search(query, k=k)

    logger.info(f"Found {len(results)} documents for query: {query}")

    return results


def clear_vector_store() -> None:
    """
    Clear all documents from vector store
    """
    global _vector_store

    logger.warning("Clearing vector store")

    if os.path.exists(CHROMA_PERSIST_DIR):
        import shutil
        shutil.rmtree(CHROMA_PERSIST_DIR)

    _vector_store = None

    logger.info("Vector store cleared")
