#!/bin/bash

# OpenAI to Ollama Proxy Startup Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 OpenAI to Ollama Proxy Startup${NC}"
echo "=================================="

# Check if Ollama is running
echo -e "${YELLOW}Checking Ollama connection...${NC}"
OLLAMA_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"

if curl -s "${OLLAMA_URL}/api/version" > /dev/null; then
    echo -e "${GREEN}✅ Ollama is running at ${OLLAMA_URL}${NC}"
else
    echo -e "${RED}❌ Ollama not found at ${OLLAMA_URL}${NC}"
    echo "Please ensure Ollama is running. You can start it with:"
    echo "  ollama serve"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate

# Install dependencies
echo -e "${YELLOW}Installing dependencies...${NC}"
pip install -r requirements.txt

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}Creating .env file from template...${NC}"
    cp .env.example .env
fi

# Start the proxy
echo -e "${GREEN}🔥 Starting OpenAI to Ollama Proxy...${NC}"
echo "The proxy will be available at: http://localhost:${PORT:-8889}"
echo "Health check: http://localhost:${PORT:-8889}/health"
echo "Press Ctrl+C to stop"
echo ""

python main.py
