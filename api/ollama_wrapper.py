"""
Ollama-compatible API wrapper for O!Store ReAct Agent
"""

import sys
from pathlib import Path
import logging
import json
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import traceback

# Add app to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.agent import OStoreAgent
from app.core.config import settings
from .models import ChatRequest, ModelInfo, TagsResponse, VersionResponse
from .streaming import async_chat_stream, create_non_stream_response

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class APIState:
    """Global state for the API wrapper"""
    def __init__(self):
        self.agent: OStoreAgent = None
        self.model_name = "ostore-agent"


api_state = APIState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("🚀 Starting O!Store Ollama API Wrapper...")

    try:
        # Initialize O!Store agent
        logger.info("🤖 Initializing O!Store ReAct Agent...")
        api_state.agent = OStoreAgent()

        # Health check
        health = api_state.agent.health_check()
        if not health["vector_stores"]:
            logger.error("❌ Vector stores not accessible")
            raise RuntimeError("Vector stores not accessible")

        logger.info("✅ O!Store Agent initialized successfully")
        logger.info(f"📊 Vector stores: {health['vector_stores']}")
        logger.info(f"🔑 OpenAI configured: {health['openai_configured']}")

        yield

    except Exception as e:
        logger.error(f"❌ Failed to initialize agent: {e}")
        raise

    finally:
        logger.info("🛑 Shutting down O!Store API...")


# Create FastAPI app
app = FastAPI(
    title="O!Store Agent API",
    description="Ollama-compatible API wrapper for O!Store ReAct Agent",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware for OpenWebUI compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests for debugging"""
    start_time = datetime.now()

    # Log request
    body = b""
    if request.method in ["POST", "PUT", "PATCH"]:
        try:
            body = await request.body()
            # Reset request body for downstream handlers
            request._body = body
        except:
            pass

    logger.info(f"📨 {request.method} {request.url.path} - Headers: {dict(request.headers)}")
    if body:
        logger.info(f"📄 Request body: {body.decode('utf-8')[:200]}...")

    # Process request
    try:
        response = await call_next(request)
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"✅ {request.method} {request.url.path} - {response.status_code} ({duration:.3f}s)")
        return response
    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        logger.error(f"❌ {request.method} {request.url.path} - ERROR ({duration:.3f}s): {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise


@app.get("/")
@app.head("/")
async def root():
    """Root endpoint - supports both GET and HEAD for Ollama compatibility"""
    return {
        "message": "O!Store Agent API",
        "version": "1.0.0",
        "endpoints": ["/api/version", "/api/tags", "/api/ps", "/api/show", "/api/generate", "/api/chat"]
    }


@app.get("/api/version")
@app.head("/api/version")
async def get_version():
    """Get version information - Ollama compatibility"""
    return VersionResponse(version="0.1.32")


@app.get("/api/tags")
@app.head("/api/tags")
async def get_tags():
    """Get available models - Ollama compatibility"""
    model_info_base = ModelInfo(
        name=api_state.model_name,
        model=api_state.model_name,
        modified_at=datetime.now(timezone.utc).isoformat(),
        size=1000000000,  # 1GB placeholder
        digest="sha256:ostore-agent-digest",
        details={
            "parent_model": "",
            "format": "gguf",
            "family": "ostore",
            "families": ["ostore"],
            "parameter_size": "7B",
            "quantization_level": "Q4_0"
        }
    )

    model_info_latest = ModelInfo(
        name=f"{api_state.model_name}:latest",
        model=f"{api_state.model_name}:latest",
        modified_at=datetime.now(timezone.utc).isoformat(),
        size=1000000000,  # 1GB placeholder
        digest="sha256:ostore-agent-digest-latest",
        details={
            "parent_model": "",
            "format": "gguf",
            "family": "ostore",
            "families": ["ostore"],
            "parameter_size": "7B",
            "quantization_level": "Q4_0"
        }
    )

    return TagsResponse(models=[model_info_base, model_info_latest])


@app.get("/v1/models")
async def get_v1_models():
    """OpenAI-compatible models endpoint for OpenWebUI"""
    return {
        "object": "list",
        "data": [
            {
                "id": api_state.model_name,
                "object": "model",
                "created": int(datetime.now(timezone.utc).timestamp()),
                "owned_by": "ostore"
            },
            {
                "id": f"{api_state.model_name}:latest",
                "object": "model",
                "created": int(datetime.now(timezone.utc).timestamp()),
                "owned_by": "ostore"
            }
        ]
    }


@app.get("/api/ps")
async def get_running_models():
    """Get running models - Ollama compatibility"""
    # Always show our model as running since it's stateless
    running_model = {
        "name": api_state.model_name,
        "model": api_state.model_name,
        "size": 1000000000,
        "digest": "sha256:ostore-agent-digest",
        "details": {
            "parent_model": "",
            "format": "gguf",
            "family": "ostore",
            "families": ["ostore"],
            "parameter_size": "7B",
            "quantization_level": "Q4_0"
        },
        "expires_at": datetime.now(timezone.utc).isoformat(),
        "size_vram": 500000000
    }

    return {"models": [running_model]}


@app.post("/api/show")
async def show_model(request: Request):
    """Show model information - Ollama compatibility"""
    try:
        payload = await request.json()
        model_name = payload.get("name", api_state.model_name)

        # Handle both "ostore-agent" and "ostore-agent:latest"
        if model_name not in [api_state.model_name, f"{api_state.model_name}:latest"]:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

        model_info = {
            "modelfile": f"# O!Store Agent\nFROM ostore-agent\nSYSTEM Ты — виртуальный консультант O!Store",
            "parameters": "temperature 0.7\ntop_p 0.9\nrepeat_penalty 1.1",
            "template": "{{ .System }}\n\n{{ .Prompt }}",
            "details": {
                "parent_model": "",
                "format": "gguf",
                "family": "ostore",
                "families": ["ostore"],
                "parameter_size": "7B",
                "quantization_level": "Q4_0"
            }
        }

        return model_info

    except Exception as e:
        logger.error(f"❌ Show model error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/generate")
async def generate(request: Request):
    """Generate endpoint for ollama run command - Ollama compatibility"""
    try:
        payload = await request.json()
        logger.info(f"📥 Generate request: {payload.get('model', 'unknown')}")

        model = payload.get("model", api_state.model_name)
        prompt = payload.get("prompt", "")
        stream = payload.get("stream", True)

        # Handle both "ostore-agent" and "ostore-agent:latest"
        if model not in [api_state.model_name, f"{api_state.model_name}:latest"]:
            raise HTTPException(status_code=404, detail=f"Model {model} not found")

        # Handle empty prompt (for ollama run initialization)
        if not prompt:
            logger.info("🔄 Empty prompt - sending welcome message")
            agent_response = "Привет! Я виртуальный консультант O!Store. Могу помочь с выбором мобильных телефонов, рассказать о характеристиках и ценах. Чем могу быть полезен?"
        else:
            logger.info(f"🗣️ Generate prompt: '{prompt[:50]}...'")

            # Generate thread ID for conversation continuity
            thread_id = f"ollama_{hash(prompt + str(datetime.now().timestamp()))}"

            # Call O!Store agent
            try:
                agent_response = api_state.agent.chat(prompt, thread_id=thread_id)
                logger.info(f"🤖 Agent response length: {len(agent_response)} chars")
            except Exception as e:
                logger.error(f"❌ Agent error: {e}")
                agent_response = f"Извините, произошла ошибка: {str(e)}"

        if stream:
            # Streaming response for ollama run
            async def generate_stream():
                try:
                    words = agent_response.split()
                    chunk_size = 3

                    for i in range(0, len(words), chunk_size):
                        chunk_words = words[i:i + chunk_size]
                        chunk_content = " ".join(chunk_words)

                        # Add trailing space if not the last chunk
                        if i + chunk_size < len(words):
                            chunk_content += " "

                        response_chunk = {
                            "model": model,
                            "created_at": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
                            "response": chunk_content,
                            "done": False
                        }

                        yield f"{json.dumps(response_chunk, ensure_ascii=False)}\n"
                        await asyncio.sleep(0.1)  # Small delay for natural typing effect

                    # Final chunk
                    final_chunk = {
                        "model": model,
                        "created_at": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
                        "response": "",
                        "done": True,
                        "total_duration": 1000000000,  # 1 second in nanoseconds
                        "load_duration": 100000000,
                        "prompt_eval_count": len(prompt.split()) if prompt else 0,
                        "prompt_eval_duration": 200000000,
                        "eval_count": len(agent_response.split()),
                        "eval_duration": 700000000
                    }
                    yield f"{json.dumps(final_chunk, ensure_ascii=False)}\n"
                except Exception as e:
                    logger.error(f"❌ Streaming error: {e}")
                    error_chunk = {
                        "model": model,
                        "created_at": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
                        "response": f"Error: {str(e)}",
                        "done": True
                    }
                    yield f"{json.dumps(error_chunk, ensure_ascii=False)}\n"

            return StreamingResponse(
                generate_stream(),
                media_type="application/x-ndjson",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                }
            )
        else:
            # Non-streaming response
            response = {
                "model": model,
                "created_at": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
                "response": agent_response,
                "done": True,
                "total_duration": 1000000000,
                "load_duration": 100000000,
                "prompt_eval_count": len(prompt.split()) if prompt else 0,
                "prompt_eval_duration": 200000000,
                "eval_count": len(agent_response.split()),
                "eval_duration": 700000000
            }
            return JSONResponse(content=response)

    except ValueError as e:
        logger.error(f"❌ Generate request parsing error: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid request: {e}")
    except Exception as e:
        logger.error(f"❌ Generate unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/api/pull")
async def pull_model(request: Request):
    """Pull model endpoint - Ollama compatibility (no-op for our case)"""
    try:
        payload = await request.json()
        model_name = payload.get("name", api_state.model_name)

        logger.info(f"📥 Pull request for model: {model_name}")

        # Handle both "ostore-agent" and "ostore-agent:latest"
        if model_name not in [api_state.model_name, f"{api_state.model_name}:latest"]:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

        # Since our model is always "available", return success immediately
        response = {
            "status": "success",
            "digest": "sha256:ostore-agent-digest",
            "total": 1000000000,
            "completed": 1000000000
        }

        return JSONResponse(content=response)

    except Exception as e:
        logger.error(f"❌ Pull model error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/chat")
async def chat(request: Request):
    """Main chat endpoint - Ollama compatibility"""
    try:
        # Parse request
        payload = await request.json()
        logger.info(f"📥 Chat request: {payload.get('model', 'unknown')} - {len(payload.get('messages', []))} messages")

        chat_request = ChatRequest.parse_obj(payload)

        if not chat_request.messages:
            raise HTTPException(status_code=400, detail="Messages list cannot be empty")

        # Extract user query
        user_message = chat_request.messages[-1]
        if user_message.role != "user":
            raise HTTPException(status_code=400, detail="Last message must be from user")

        user_query = user_message.content

        # Generate thread ID from messages for conversation continuity
        thread_id = f"ostore_{hash(''.join([msg.content for msg in chat_request.messages]))}"

        logger.info(f"🗣️ User query: '{user_query[:50]}...'")
        logger.info(f"🧵 Thread ID: {thread_id}")

        # Call O!Store agent (keeping all logic intact)
        try:
            agent_response = api_state.agent.chat(user_query, thread_id=thread_id)
            logger.info(f"🤖 Agent response length: {len(agent_response)} chars")

        except Exception as e:
            logger.error(f"❌ Agent error: {e}")
            agent_response = f"Извините, произошла ошибка при обработке запроса: {str(e)}"

        # Return streaming or non-streaming response
        if chat_request.stream:
            logger.info("📡 Returning streaming response")

            # For Ollama CLI compatibility, return generate-style format (plain JSON lines)
            # instead of chat-style format (SSE with data: prefix)
            async def ollama_chat_stream():
                try:
                    words = agent_response.split()
                    chunk_size = 3

                    for i in range(0, len(words), chunk_size):
                        chunk_words = words[i:i + chunk_size]
                        chunk_content = " ".join(chunk_words)

                        # Add trailing space if not the last chunk
                        if i + chunk_size < len(words):
                            chunk_content += " "

                        response_chunk = {
                            "model": api_state.model_name,
                            "created_at": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
                            "message": {
                                "role": "assistant",
                                "content": chunk_content
                            },
                            "done": False
                        }

                        yield f"{json.dumps(response_chunk, ensure_ascii=False)}\n"
                        await asyncio.sleep(0.1)

                    # Final chunk
                    final_chunk = {
                        "model": api_state.model_name,
                        "created_at": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
                        "message": {
                            "role": "assistant",
                            "content": ""
                        },
                        "done": True,
                        "done_reason": "stop"
                    }
                    yield f"{json.dumps(final_chunk, ensure_ascii=False)}\n"
                except Exception as e:
                    logger.error(f"❌ Chat streaming error: {e}")
                    error_chunk = {
                        "model": api_state.model_name,
                        "created_at": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
                        "message": {
                            "role": "assistant",
                            "content": f"Error: {str(e)}"
                        },
                        "done": True
                    }
                    yield f"{json.dumps(error_chunk, ensure_ascii=False)}\n"

            return StreamingResponse(
                ollama_chat_stream(),
                media_type="application/x-ndjson",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                }
            )
        else:
            logger.info("📄 Returning non-streaming response")
            response = create_non_stream_response(agent_response, api_state.model_name)
            return JSONResponse(content=response)

    except ValueError as e:
        logger.error(f"❌ Request parsing error: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid request: {e}")

    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if not api_state.agent:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "reason": "Agent not initialized"}
        )

    health = api_state.agent.health_check()
    stats = api_state.agent.logger.get_session_stats()

    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "agent": {
            "vector_stores": health["vector_stores"],
            "openai_configured": health["openai_configured"]
        },
        "analytics": {
            "total_sessions": stats.get("total_sessions", 0),
            "total_queries": stats.get("total_queries", 0)
        }
    }


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": f"Path {request.url.path} not found"}
    )


@app.exception_handler(500)
async def server_error_handler(request: Request, exc):
    logger.error(f"Server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )


if __name__ == "__main__":
    import uvicorn

    logger.info("🚀 Starting O!Store Ollama API Wrapper...")
    uvicorn.run(
        "ollama_wrapper:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )