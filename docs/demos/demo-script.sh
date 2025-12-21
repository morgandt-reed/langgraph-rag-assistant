#!/bin/bash
# Demo Script for langgraph-rag-assistant
#
# This script demonstrates the RAG question-answering flow.
# Record with asciinema or OBS, convert to GIF with gifski.
#
# Duration target: 30 seconds

set -e

echo "=== LangGraph RAG Assistant Demo ==="
echo ""

# Step 1: Show the service is running
echo "🚀 Starting RAG Assistant..."
docker-compose up -d
sleep 3

# Step 2: Check health
echo ""
echo "✅ Service health check:"
curl -s http://localhost:8000/health | jq .
sleep 1

# Step 3: Ingest a sample document (if not already done)
echo ""
echo "📄 Ingesting sample document..."
curl -s -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"source": "sample", "content": "Docker is a platform for developing, shipping, and running applications in containers. Containers are lightweight, portable, and ensure consistency across environments."}' | jq .
sleep 2

# Step 4: Ask a question
echo ""
echo "❓ Asking: 'What is Docker and why use containers?'"
echo ""
curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Docker and why use containers?"}' | jq .
sleep 3

# Step 5: Show sources
echo ""
echo "📚 The response includes source attribution and confidence scores."

# Step 6: Ask follow-up (demonstrates conversation memory)
echo ""
echo "❓ Follow-up: 'How do they ensure consistency?'"
curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How do they ensure consistency?", "session_id": "demo"}' | jq .

echo ""
echo "✅ Demo complete!"
echo ""
echo "Features demonstrated:"
echo "  - Document ingestion"
echo "  - Semantic search and retrieval"
echo "  - LLM-powered answer generation"
echo "  - Source attribution"
echo "  - Conversation memory"
