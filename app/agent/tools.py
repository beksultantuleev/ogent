"""
Retriever tools for the ReAct agent
"""

import os

os.environ["HTTP_PROXY"] = "http://172.27.129.0:3128"
os.environ["HTTPS_PROXY"] = "http://172.27.129.0:3128"

from langchain.tools import Tool
from langchain_core.vectorstores import VectorStoreRetriever

from ..utils.logging import AnalyticsLogger


def get_retriever_tools(specs_retriever: VectorStoreRetriever,
                       docs_retriever: VectorStoreRetriever) -> list[Tool]:
    """
    Create retriever tools for the agent

    Args:
        specs_retriever: Mobile phone specifications retriever
        docs_retriever: Store documentation retriever

    Returns:
        List of langchain Tools
    """
    logger = AnalyticsLogger()

    def specs_tool_func(query: str) -> str:
        """Query mobile phone specifications"""
        docs = specs_retriever.invoke(query)
        logger.log_query("specs_retriever", query, len(docs))
        return "\n\n".join(doc.page_content for doc in docs)

    def docs_tool_func(query: str) -> str:
        """Query store documentation"""
        docs = docs_retriever.invoke(query)
        logger.log_query("docs_retriever", query, len(docs))
        return "\n\n".join(doc.page_content for doc in docs)

    specs_tool = Tool(
        name="specs_retriever",
        func=specs_tool_func,
        description=(
            "Возвращает информацию о телефонах в JSON с ключевыми характеристиками: "
            "Используй этот инструмент для получения подробной информации о мобильных "
            "телефонах, включая бренд, модель, вес, объем RAM, характеристики фронтальной "
            "и основной камеры, процессор, емкость батареи, размер экрана, цену при запуске и год выпуска."
        )
    )

    docs_tool = Tool(
        name="docs_retriever",
        func=docs_tool_func,
        description=(
            "Используйте этот инструмент для получения подробной информации о магазине: "
            "гарантии, адреса и контакты всех точек (главный офис, филиалы), телефоны, e-mail, "
            "график работы, условия рассрочки, FAQ, доставка, текущие акции и скидки, официальные документы."
        )
    )

    return [specs_tool, docs_tool]