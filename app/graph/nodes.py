from typing import Dict, Any
import logging
from langchain_openai import ChatOpenAI
from .state import GraphState, Document

logger = logging.getLogger(__name__)


def query_analysis_node(state: GraphState) -> Dict[str, Any]:
    """
    Analyze the user query to determine next steps
    """
    logger.info("Executing query analysis node")

    question = state["question"]

    # Simple heuristics (in production, use LLM for classification)
    needs_retrieval = True  # Most questions need retrieval
    needs_clarification = False

    # Check if question is too vague
    if len(question.strip().split()) < 3:
        needs_clarification = True

    state["needs_retrieval"] = needs_retrieval
    state["needs_clarification"] = needs_clarification
    state["steps_taken"] = state.get("steps_taken", []) + ["query_analysis"]

    logger.info(f"Needs retrieval: {needs_retrieval}, Needs clarification: {needs_clarification}")

    return state


def retrieval_node(state: GraphState) -> Dict[str, Any]:
    """
    Retrieve relevant documents from vector store
    """
    logger.info("Executing retrieval node")

    question = state["question"]

    try:
        from ..retrieval.vector_store import get_vector_store

        vector_store = get_vector_store()

        # Perform similarity search
        results = vector_store.similarity_search_with_score(
            query=question,
            k=5
        )

        # Convert to Document objects
        documents = []
        for doc, score in results:
            documents.append(Document(
                content=doc.page_content,
                metadata=doc.metadata,
                relevance_score=float(score)
            ))

        state["retrieved_documents"] = documents
        state["steps_taken"] = state.get("steps_taken", []) + ["retrieval"]

        logger.info(f"Retrieved {len(documents)} documents")

    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        state["retrieved_documents"] = []
        state["error"] = str(e)

    return state


def relevance_check_node(state: GraphState) -> Dict[str, Any]:
    """
    Check if retrieved documents are relevant enough
    """
    logger.info("Executing relevance check node")

    documents = state.get("retrieved_documents", [])

    if not documents:
        state["confidence"] = 0.0
        logger.info("No documents retrieved, confidence: 0.0")
    else:
        # Calculate average relevance score
        avg_score = sum(doc.relevance_score or 0 for doc in documents) / len(documents)
        state["confidence"] = min(avg_score, 1.0)
        logger.info(f"Average relevance score: {avg_score:.2f}")

    state["steps_taken"] = state.get("steps_taken", []) + ["relevance_check"]

    return state


def generation_node(state: GraphState) -> Dict[str, Any]:
    """
    Generate answer using LLM with retrieved context
    """
    logger.info("Executing generation node")

    question = state["question"]
    documents = state.get("retrieved_documents", [])

    try:
        # Prepare context from retrieved documents
        context = "\n\n".join([
            f"Document {i+1} (Source: {doc.metadata.get('source', 'unknown')}):\n{doc.content}"
            for i, doc in enumerate(documents)
        ])

        # Create prompt
        system_prompt = """You are a helpful technical documentation assistant.
Answer questions based ONLY on the provided context.
If the context doesn't contain relevant information, say so clearly.
Always cite specific sources when providing information."""

        user_prompt = f"""Context:
{context}

Question: {question}

Answer the question based on the context above. Be specific and cite sources."""

        # Initialize LLM
        llm = ChatOpenAI(
            model="gpt-4-turbo-preview",
            temperature=0.1
        )

        # Generate response
        response = llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])

        state["answer"] = response.content
        state["steps_taken"] = state.get("steps_taken", []) + ["generation"]

        logger.info("Answer generated successfully")

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        state["answer"] = "I apologize, but I encountered an error while generating the answer."
        state["error"] = str(e)
        state["confidence"] = 0.0

    return state


def source_attribution_node(state: GraphState) -> Dict[str, Any]:
    """
    Extract and format source citations
    """
    logger.info("Executing source attribution node")

    documents = state.get("retrieved_documents", [])

    sources = []
    for i, doc in enumerate(documents[:3]):  # Top 3 sources
        source_info = {
            "document": doc.metadata.get("source", f"Document {i+1}"),
            "page": doc.metadata.get("page", None),
            "relevance_score": round(doc.relevance_score or 0.0, 2),
            "excerpt": doc.content[:200] + "..." if len(doc.content) > 200 else doc.content
        }
        sources.append(source_info)

    state["sources"] = sources
    state["steps_taken"] = state.get("steps_taken", []) + ["source_attribution"]

    logger.info(f"Added {len(sources)} source citations")

    return state


def fallback_node(state: GraphState) -> Dict[str, Any]:
    """
    Provide fallback response when no relevant documents found
    """
    logger.info("Executing fallback node")

    state["answer"] = """I apologize, but I couldn't find relevant information in the documentation to answer your question.

This could be because:
- The topic is not covered in the available documentation
- Your question might need rephrasing for better results
- The information might be in a different section

Could you please rephrase your question or provide more context?"""

    state["confidence"] = 0.0
    state["sources"] = []
    state["steps_taken"] = state.get("steps_taken", []) + ["fallback"]

    return state


def clarification_node(state: GraphState) -> Dict[str, Any]:
    """
    Request clarification from user
    """
    logger.info("Executing clarification node")

    state["clarification_question"] = "Could you please provide more details or rephrase your question?"
    state["steps_taken"] = state.get("steps_taken", []) + ["clarification"]

    return state
