"""
Vector store management for O!Store Agent
"""
import os

os.environ["HTTP_PROXY"] = "http://172.27.129.0:3128"
os.environ["HTTPS_PROXY"] = "http://172.27.129.0:3128"
from typing import Tuple

from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.vectorstores import VectorStoreRetriever
from qdrant_client import QdrantClient

from .config import settings


class VectorStoreManager:
    """Manages vector store connections and retrievers"""

    def __init__(self):
        self.client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port
        )
        self.embeddings = OpenAIEmbeddings(
            model=settings.embedding_model
        )

    def get_retrievers(self) -> Tuple[VectorStoreRetriever, VectorStoreRetriever]:
        """
        Get specs and docs retrievers

        Returns:
            Tuple of (specs_retriever, docs_retriever)
        """
        # Specs retriever
        specs_store = QdrantVectorStore(
            client=self.client,
            collection_name=settings.specs_collection,
            embedding=self.embeddings
        )
        specs_retriever = specs_store.as_retriever(
            search_kwargs={"k": settings.retriever_k}
        )

        # Docs retriever
        docs_store = QdrantVectorStore(
            client=self.client,
            collection_name=settings.docs_collection,
            embedding=self.embeddings
        )
        docs_retriever = docs_store.as_retriever(
            search_kwargs={"k": settings.retriever_k}
        )

        return specs_retriever, docs_retriever

    def health_check(self) -> bool:
        """Check if vector stores are accessible"""
        try:
            collections = self.client.get_collections()
            collection_names = [c.name for c in collections.collections]

            specs_exists = settings.specs_collection in collection_names
            docs_exists = settings.docs_collection in collection_names

            return specs_exists and docs_exists
        except Exception:
            return False