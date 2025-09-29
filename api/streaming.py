"""
Streaming response handler for Ollama API compatibility
"""

import json
from datetime import datetime, timezone
from typing import Generator, AsyncGenerator
import asyncio


def create_chat_stream(response_content: str, model: str) -> Generator[str, None, None]:
    """
    Create a streaming response in Ollama format

    Args:
        response_content: The complete response from O!Store agent
        model: Model name to include in response

    Yields:
        JSON strings in Ollama streaming format
    """
    # Split response into chunks for streaming effect
    words = response_content.split()
    chunk_size = 3  # Send 3 words at a time

    for i in range(0, len(words), chunk_size):
        chunk_words = words[i:i + chunk_size]
        chunk_content = " ".join(chunk_words)

        # Add space if not the last chunk
        if i + chunk_size < len(words):
            chunk_content += " "

        response_chunk = {
            "model": model,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "message": {
                "role": "assistant",
                "content": chunk_content
            },
            "done": False
        }

        yield f"data: {json.dumps(response_chunk, ensure_ascii=False)}\n\n"

    # Send final chunk marking completion
    final_chunk = {
        "model": model,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "message": {
            "role": "assistant",
            "content": ""
        },
        "done": True,
        "done_reason": "stop"
    }

    yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"


async def async_chat_stream(response_content: str, model: str) -> AsyncGenerator[str, None]:
    """
    Async version of create_chat_stream with small delays for realistic streaming

    Args:
        response_content: The complete response from O!Store agent
        model: Model name to include in response

    Yields:
        JSON strings in Ollama streaming format
    """
    words = response_content.split()
    chunk_size = 3

    for i in range(0, len(words), chunk_size):
        chunk_words = words[i:i + chunk_size]
        chunk_content = " ".join(chunk_words)

        if i + chunk_size < len(words):
            chunk_content += " "

        response_chunk = {
            "model": model,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "message": {
                "role": "assistant",
                "content": chunk_content
            },
            "done": False
        }

        yield f"data: {json.dumps(response_chunk, ensure_ascii=False)}\n\n"

        # Small delay for realistic streaming
        await asyncio.sleep(0.05)

    # Final completion chunk
    final_chunk = {
        "model": model,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "message": {
            "role": "assistant",
            "content": ""
        },
        "done": True,
        "done_reason": "stop"
    }

    yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"


def create_non_stream_response(response_content: str, model: str) -> dict:
    """
    Create a non-streaming response in Ollama format

    Args:
        response_content: The complete response from O!Store agent
        model: Model name to include in response

    Returns:
        Dictionary in Ollama response format
    """
    return {
        "model": model,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "message": {
            "role": "assistant",
            "content": response_content
        },
        "done": True,
        "done_reason": "stop"
    }