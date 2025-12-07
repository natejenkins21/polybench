#!/bin/bash
# Start the FastAPI server

echo "🚀 Starting PolyBench API server..."
echo "   API will be available at: http://localhost:8000"
echo "   API docs at: http://localhost:8000/docs"
echo ""

# Load environment variables if .env exists
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Start the API server
python3.11 -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload

