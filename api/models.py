"""
Pydantic models for Ollama API compatibility
"""

from typing import Optional, Dict, Union, List, Any
from pydantic import BaseModel


class Message(BaseModel):
    """Chat message model"""
    role: str
    content: str
    images: Optional[List[str]] = None


class ChatRequest(BaseModel):
    """Chat request model compatible with Ollama API"""
    model: str
    messages: List[Message]
    stream: Optional[bool] = True
    format: Optional[str] = None
    options: Optional[Dict[str, Any]] = None
    keep_alive: Optional[Union[str, int]] = None

    class Config:
        extra = "allow"  # Accept extra unexpected fields


class ChatResponse(BaseModel):
    """Chat response model"""
    model: str
    created_at: str
    message: Message
    done: bool
    done_reason: Optional[str] = None


class ModelInfo(BaseModel):
    """Model information for /api/tags endpoint"""
    name: str
    model: str
    modified_at: str
    size: int
    digest: str
    details: Dict[str, Any]


class TagsResponse(BaseModel):
    """Response for /api/tags endpoint"""
    models: List[ModelInfo]


class VersionResponse(BaseModel):
    """Response for /api/version endpoint"""
    version: str