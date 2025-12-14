import os
from typing import List
import logging
from langchain.schema import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader
)
from .chunker import chunk_documents

logger = logging.getLogger(__name__)


class DocumentLoader:
    """
    Load documents from various formats
    """

    def __init__(self, docs_directory: str = "./sample-docs/technical-docs"):
        self.docs_directory = docs_directory

    def load_directory(self) -> List[Document]:
        """
        Load all documents from directory
        """
        logger.info(f"Loading documents from {self.docs_directory}")

        all_documents = []

        # Load PDFs
        try:
            pdf_loader = DirectoryLoader(
                self.docs_directory,
                glob="**/*.pdf",
                loader_cls=PyPDFLoader
            )
            pdf_docs = pdf_loader.load()
            all_documents.extend(pdf_docs)
            logger.info(f"Loaded {len(pdf_docs)} PDF documents")
        except Exception as e:
            logger.warning(f"Failed to load PDFs: {e}")

        # Load text files
        try:
            txt_loader = DirectoryLoader(
                self.docs_directory,
                glob="**/*.txt",
                loader_cls=TextLoader
            )
            txt_docs = txt_loader.load()
            all_documents.extend(txt_docs)
            logger.info(f"Loaded {len(txt_docs)} text documents")
        except Exception as e:
            logger.warning(f"Failed to load text files: {e}")

        # Load markdown files
        try:
            md_loader = DirectoryLoader(
                self.docs_directory,
                glob="**/*.md",
                loader_cls=TextLoader
            )
            md_docs = md_loader.load()
            all_documents.extend(md_docs)
            logger.info(f"Loaded {len(md_docs)} markdown documents")
        except Exception as e:
            logger.warning(f"Failed to load markdown files: {e}")

        logger.info(f"Total documents loaded: {len(all_documents)}")

        return all_documents

    def load_and_chunk(self) -> List[Document]:
        """
        Load documents and chunk them for ingestion
        """
        documents = self.load_directory()

        if not documents:
            logger.warning("No documents found to load")
            return []

        # Chunk documents
        chunked_docs = chunk_documents(documents)

        logger.info(f"Created {len(chunked_docs)} chunks from {len(documents)} documents")

        return chunked_docs


def load_sample_documents() -> List[Document]:
    """
    Load sample documents for testing
    """
    sample_docs = [
        Document(
            page_content="""Docker is a containerization platform that allows developers to package
            applications with all their dependencies into standardized units called containers.
            Containers are lightweight, portable, and ensure consistency across different environments.""",
            metadata={"source": "docker-intro.txt", "page": 1}
        ),
        Document(
            page_content="""To deploy a Docker container, you first need to create a Dockerfile that
            defines your application's environment. Then use 'docker build' to create an image, and
            'docker run' to start a container from that image.""",
            metadata={"source": "docker-deployment.txt", "page": 1}
        ),
        Document(
            page_content="""Kubernetes is an orchestration platform for managing containerized applications
            at scale. It provides features like automatic scaling, load balancing, self-healing, and
            rolling updates.""",
            metadata={"source": "kubernetes-intro.txt", "page": 1}
        )
    ]

    logger.info(f"Loaded {len(sample_docs)} sample documents")

    return sample_docs
