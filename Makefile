.PHONY: help install start test clean

# Default target
help:
	@echo "OpenAI to Ollama Proxy - Available Commands:"
	@echo ""
	@echo "  install     - Install dependencies"
	@echo "  start       - Start the proxy server"
	@echo "  test        - Run the test suite"
	@echo "  example     - Run the example client"
	@echo "  clean       - Clean up temporary files"
	@echo ""

# Install dependencies
install:
	pip install -r requirements.txt

# Start the proxy server
start:
	./start.sh

# Run tests
test:
	python test_proxy.py

# Run example client
example:
	python example_client.py

# Clean up
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
