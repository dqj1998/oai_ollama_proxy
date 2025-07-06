# OpenAI to Ollama Proxy

A FastAPI-based proxy server that bridges OpenAI-compatible API calls to Ollama, allowing you to use OpenAI client libraries with locally-hosted Ollama models.

Try to play with your local LLMs with a typical config of Cline, even Cline's prompts are too complex for most local LLMs:
<img width="547" alt="oai_ollama_cline" src="https://github.com/user-attachments/assets/aacd076f-82a7-496b-83ae-eb820e52eff7" />


## Features

- 🔄 **OpenAI API Compatibility**: Drop-in replacement for OpenAI's chat completions API
- 🚀 **Streaming Support**: Real-time streaming responses
- 📋 **Model Management**: List and use any available Ollama models
-  **Easy Configuration**: Environment-based configuration
- 📊 **Health Monitoring**: Built-in health check endpoints
- 🧪 **Comprehensive Testing**: Full test suite included

## Quick Start

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai/) installed and running

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd oai_ollama_proxy
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment (optional):
```bash
cp .env.example .env
# Edit .env with your settings
```

4. Start the proxy:
```bash
python main.py
```

The proxy will start on `http://localhost:8889` by default.

## Usage

### With OpenAI Python Library

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8889/v1",
    api_key="not-needed"  # Required by client, but not used
)

# Chat completion
response = client.chat.completions.create(
    model="llama2",
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)

print(response.choices[0].message.content)
```

### With cURL

```bash
# List models
curl http://localhost:8889/v1/models

# Chat completion
curl -X POST http://localhost:8889/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama2",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": false
  }'

# Streaming chat
curl -X POST http://localhost:8889/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama2",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

## API Endpoints

### Core Endpoints

- `GET /` - Root endpoint with service info
- `GET /health` - Health check endpoint
- `GET /v1/models` - List available Ollama models
- `POST /v1/chat/completions` - Create chat completions (streaming and non-streaming)

### Supported Parameters

The proxy supports most OpenAI chat completion parameters:

- `model` - Ollama model name
- `messages` - Conversation messages
- `max_tokens` - Maximum tokens to generate
- `temperature` - Sampling temperature (0.0 to 1.0)
- `top_p` - Nucleus sampling parameter
- `stream` - Enable streaming responses
- `stop` - Stop sequences

## Configuration

Configure the proxy using environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `HOST` | `0.0.0.0` | Proxy server host |
| `PORT` | `8000` | Proxy server port |
| `DEFAULT_MODEL` | `llama2` | Default model name |

## Testing

Run the comprehensive test suite:

```bash
python test_proxy.py
```

Run the example client:

```bash
# Install OpenAI library first
pip install openai

python example_client.py
```

## Development

### Project Structure

```
oai_ollama_proxy/
├── main.py              # Main proxy server
├── test_proxy.py        # Test suite
├── example_client.py    # Example usage
├── requirements.txt     # Python dependencies
├── start.sh            # Startup script
├── Makefile            # Common tasks
├── .env.example        # Environment template
└── README.md           # This file
```

### Running in Development

1. Start Ollama:
```bash
ollama serve
```

2. Install a model:
```bash
ollama pull llama2
```

3. Start the proxy:
```bash
python main.py
```

4. Test the proxy:
```bash
python test_proxy.py
```

## OpenAI Compatibility

The proxy translates between OpenAI and Ollama APIs:

| OpenAI Parameter | Ollama Equivalent | Notes |
|------------------|-------------------|-------|
| `model` | `model` | Direct mapping |
| `messages` | `prompt` | Converted to text prompt |
| `max_tokens` | `num_predict` | Token limit |
| `temperature` | `temperature` | Sampling temperature |
| `top_p` | `top_p` | Nucleus sampling |
| `stream` | `stream` | Streaming mode |
| `stop` | `stop` | Stop sequences |

## Troubleshooting

### Common Issues

1. **Connection refused**: Ensure Ollama is running on the configured URL
2. **Model not found**: Pull the model with `ollama pull <model-name>`
3. **Timeout errors**: Increase timeout values for large models

### Health Check

Check the proxy status:
```bash
curl http://localhost:8889/health
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `python test_proxy.py`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Ollama](https://ollama.ai/) for the excellent local LLM platform
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [OpenAI](https://openai.com/) for the API specification
