from typing import List
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

logger = logging.getLogger(__name__)


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> List[Document]:
    """
    Split documents into smaller chunks for better retrieval
    """
    logger.info(f"Chunking {len(documents)} documents")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len
    )

    chunked_docs = text_splitter.split_documents(documents)

    logger.info(f"Created {len(chunked_docs)} chunks")

    return chunked_docs
