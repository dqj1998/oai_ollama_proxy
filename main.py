import os
from typing import List, Optional, Dict, Any, AsyncGenerator, Union
import httpx
import json
import time
import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError
import uvicorn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
log_file = os.getenv("LOG_FILE", None)

# Setup logging handlers
handlers = [logging.StreamHandler()]  # Console output
if log_file:
    handlers.append(logging.FileHandler(log_file))

logging.basicConfig(
    level=getattr(logging, log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=handlers
)
logger = logging.getLogger("oai_ollama_proxy")

app = FastAPI(
    title="OpenAI to Ollama Proxy",
    description="Bridge OpenAI compatible API calls to Ollama calls",
    version="1.0.0"
)

# Middleware to log all requests
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests with full details"""
    start_time = time.time()
    
    # Log request details
    logger.info("📨 Incoming Request:")
    logger.info(f"   Method: {request.method}")
    logger.info(f"   URL: {request.url}")
    logger.info(f"   Client: {request.client.host}:{request.client.port}")
    logger.info(f"   User-Agent: {request.headers.get('user-agent', 'Unknown')}")
    logger.info(f"   Content-Type: {request.headers.get('content-type', 'None')}")
    
    if request.method == "POST":
        logger.info("   Request Body: (not logged in middleware to support streaming)")
    
    # Process the request
    response = await call_next(request)
    
    # Log response details
    process_time = time.time() - start_time
    logger.info("📤 Response:")
    logger.info(f"   Status: {response.status_code}")
    logger.info(f"   Process Time: {process_time:.3f}s")
    
    return response

# Custom exception handler for validation errors
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors with detailed information"""
    error_details = []
    for error in exc.errors():
        error_details.append({
            "field": " -> ".join(str(x) for x in error["loc"]),
            "message": error["msg"],
            "type": error["type"]
        })
    
    # Log the validation error
    logger.error("❌ Request Validation Failed:")
    logger.error(f"   URL: {request.url}")
    logger.error(f"   Method: {request.method}")
    for detail in error_details:
        logger.error(f"   Field '{detail['field']}': {detail['message']} (type: {detail['type']})")
    
    if hasattr(exc, 'body') and exc.body:
        try:
            body_str = exc.body.decode('utf-8') if isinstance(exc.body, bytes) else str(exc.body)
            logger.error(f"   Request Body: {body_str}")
        except Exception as e:
            logger.error(f"   Could not log request body: {e}")
    
    return JSONResponse(
        status_code=422,
        content={
            "error": "Request validation failed",
            "details": error_details,
            "request_body": str(exc.body) if hasattr(exc, 'body') else None
        }
    )

# Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "llama2")

# OpenAI compatible models
class ContentPart(BaseModel):
    type: str
    text: Optional[str] = None

class Message(BaseModel):
    role: str  # "system", "user", "assistant"
    content: Union[str, List[ContentPart]]

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = None
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None
    
    class Config:
        # Allow extra fields to be ignored rather than causing validation errors
        extra = "ignore"

class Choice(BaseModel):
    index: int
    message: Message
    finish_reason: str

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Usage

class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[Dict[str, Any]]

class Model(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "ollama"

class ModelsResponse(BaseModel):
    object: str = "list"
    data: List[Model]

def get_content_text(content: Union[str, List[ContentPart]]) -> str:
    """Extract text from content, whether it's a string or a list of content parts"""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        return "\n".join(part.text for part in content if part.type == "text" and part.text)
    else:
        raise ValueError("Invalid content type")

def convert_openai_to_ollama(request: ChatCompletionRequest) -> Dict[str, Any]:
    """Convert OpenAI chat completion request to Ollama format"""
    # Build the prompt from messages
    prompt = ""
    for message in request.messages:
        content_text = get_content_text(message.content)
        if message.role == "system":
            prompt += f"System: {content_text}\n"
        elif message.role == "user":
            prompt += f"User: {content_text}\n"
        elif message.role == "assistant":
            prompt += f"Assistant: {content_text}\n"
    
    prompt += "Assistant: "
    
    ollama_request = {
        "model": request.model,
        "prompt": prompt,
        "stream": request.stream,
        "options": {}
    }
    
    # Map OpenAI parameters to Ollama options
    if request.temperature is not None:
        ollama_request["options"]["temperature"] = request.temperature
    
    if request.max_tokens is not None:
        ollama_request["options"]["num_predict"] = request.max_tokens
    
    if request.top_p is not None:
        ollama_request["options"]["top_p"] = request.top_p
    
    if request.stop:
        ollama_request["options"]["stop"] = request.stop
    
    return ollama_request

def convert_ollama_to_openai(ollama_response: Dict[str, Any], model: str, request_id: str) -> ChatCompletionResponse:
    """Convert Ollama response to OpenAI format"""
    response_text = ollama_response.get("response", "")
    
    # Estimate token usage (rough approximation)
    prompt_tokens = len(ollama_response.get("prompt", "")) // 4
    completion_tokens = len(response_text) // 4
    
    return ChatCompletionResponse(
        id=request_id,
        created=int(time.time()),
        model=model,
        choices=[
            Choice(
                index=0,
                message=Message(role="assistant", content=response_text),
                finish_reason="stop"
            )
        ],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens
        )
    )

@app.get("/")
async def root():
    return {"message": "OpenAI to Ollama Proxy", "status": "running"}

@app.get("/v1/models")
async def list_models():
    """List available models from Ollama"""
    logger.info("📋 Listing available models from Ollama...")
    try:
        timeout_value = int(os.getenv('TIMEOUT', 300))
        async with httpx.AsyncClient(timeout=timeout_value) as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            logger.info(f"📥 Ollama models response status: {response.status_code}")
            
            if response.status_code == 200:
                ollama_models = response.json()
                models = []
                for model in ollama_models.get("models", []):
                    models.append(Model(
                        id=model["name"],
                        created=int(time.time())
                    ))
                logger.info(f"✅ Found {len(models)} models")
                for model in models:
                    logger.info(f"   - {model.id}")
                return ModelsResponse(data=models)
            else:
                logger.error(f"❌ Failed to fetch models from Ollama: {response.status_code}")
                raise HTTPException(status_code=500, detail="Failed to fetch models from Ollama")
    except Exception as e:
        logger.error(f"❌ Error connecting to Ollama: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error connecting to Ollama: {str(e)}")

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest, raw_request: Request):
    """Create a chat completion using Ollama"""
    request_id = f"chatcmpl-{int(time.time())}"
    
    # Log the parsed request details
    logger.info(f"🤖 Chat Completion Request (ID: {request_id}):")
    logger.info(f"   Model: {request.model}")
    logger.info(f"   Messages: {len(request.messages)} message(s)")
    for i, msg in enumerate(request.messages):
        content_text = get_content_text(msg.content)
        logger.debug(f"     Message {i}: Role - {msg.role}, Content - {msg.content}")
        logger.info(f"     [{i}] {msg.role}: {content_text[:100]}{'...' if len(content_text) > 100 else ''}")
    logger.info(f"   Stream: {request.stream}")
    logger.info(f"   Max Tokens: {request.max_tokens}")
    logger.info(f"   Temperature: {request.temperature}")
    
    try:
        # Convert OpenAI request to Ollama format
        ollama_request = convert_openai_to_ollama(request)
        logger.info("🔄 Converted to Ollama request:")
        logger.info(f"   Model: {ollama_request['model']}")
        logger.info(f"   Prompt length: {len(ollama_request['prompt'])} chars")
        logger.info(f"   Stream: {ollama_request['stream']}")
        logger.info(f"   Options: {ollama_request.get('options', {})}")
        
        if request.stream:
            logger.info("📡 Starting streaming response...")
            return StreamingResponse(
                stream_chat_completion(ollama_request, request.model, request_id),
                media_type="text/plain"
            )
        else:
            logger.info("📞 Making non-streaming request to Ollama...")
            timeout_value = int(os.getenv('TIMEOUT', 300))
            async with httpx.AsyncClient(timeout=timeout_value) as client:
                response = await client.post(
                    f"{OLLAMA_BASE_URL}/api/generate",
                    json=ollama_request
                )
                
                logger.info(f"📥 Ollama response status: {response.status_code}")
                
                if response.status_code == 200:
                    ollama_response = response.json()
                    logger.info("✅ Ollama response received:")
                    logger.info(f"   Response length: {len(ollama_response.get('response', ''))} chars")
                    logger.info(f"   Done: {ollama_response.get('done', False)}")
                    
                    openai_response = convert_ollama_to_openai(ollama_response, request.model, request_id)
                    logger.info("🔄 Converted to OpenAI format, returning response")
                    return openai_response
                else:
                    logger.error(f"❌ Ollama API error: {response.status_code}")
                    logger.error(f"   Response: {response.text}")
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=f"Ollama API error: {response.text}"
                    )
    
    except Exception as e:
        logger.error(f"❌ Error processing request: {str(e)}")
        logger.error(f"   Request ID: {request_id}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

async def stream_chat_completion(ollama_request: Dict[str, Any], model: str, request_id: str) -> AsyncGenerator[str, None]:
    """Stream chat completion responses"""
    try:
        timeout_value = int(os.getenv('TIMEOUT', 300))
        async with httpx.AsyncClient(timeout=timeout_value) as client:
            async with client.stream(
                "POST",
                f"{OLLAMA_BASE_URL}/api/generate",
                json=ollama_request
            ) as response:
                if response.status_code != 200:
                    yield f"data: {{\"error\": \"Ollama API error: {response.status_code}\"}}\n\n"
                    return
                
                async for line in response.aiter_lines():
                    if line.strip():
                        try:
                            ollama_chunk = json.loads(line)
                            
                            # Convert to OpenAI streaming format
                            chunk = ChatCompletionChunk(
                                id=request_id,
                                created=int(time.time()),
                                model=model,
                                choices=[{
                                    "index": 0,
                                    "delta": {
                                        "content": ollama_chunk.get("response", "")
                                    },
                                    "finish_reason": "stop" if ollama_chunk.get("done", False) else None
                                }]
                            )
                            
                            yield f"data: {chunk.model_dump_json()}\n\n"
                            
                            if ollama_chunk.get("done", False):
                                yield "data: [DONE]\n\n"
                                break
                                
                        except json.JSONDecodeError:
                            continue
    
    except Exception as e:
        yield f"data: {{\"error\": \"Streaming error: {str(e)}\"}}\n\n"

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    logger.info("🏥 Health check requested...")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/version", timeout=5.0)
            ollama_status = "healthy" if response.status_code == 200 else "unhealthy"
            logger.info(f"   Ollama status: {ollama_status} (HTTP {response.status_code})")
    except Exception as e:
        ollama_status = "unreachable"
        logger.warning(f"   Ollama unreachable: {str(e)}")
    
    health_response = {
        "status": "healthy",
        "ollama_status": ollama_status,
        "ollama_url": OLLAMA_BASE_URL
    }
    
    logger.info(f"✅ Health check complete: {health_response}")
    return health_response

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8889))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info("🚀 Starting OpenAI to Ollama Proxy")
    logger.info("=" * 50)
    logger.info(f"Server: {host}:{port}")
    logger.info(f"Ollama URL: {OLLAMA_BASE_URL}")
    logger.info(f"Default Model: {DEFAULT_MODEL}")
    logger.info("Endpoints:")
    logger.info(f"  Health: http://{host}:{port}/health")
    logger.info(f"  Models: http://{host}:{port}/v1/models")
    logger.info(f"  Chat: http://{host}:{port}/v1/chat/completions")
    logger.info("=" * 50)
    
    uvicorn.run(app, host=host, port=port)
