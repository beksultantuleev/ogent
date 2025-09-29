"""
O!Store ReAct Agent - Main agent implementation
"""

from datetime import datetime
from typing import Dict, Any, List

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, MessagesState, END
from langgraph.checkpoint.memory import InMemorySaver
from langmem.short_term import SummarizationNode, RunningSummary

from .tools import get_retriever_tools
from ..core.config import settings
from ..core.vectorstore import VectorStoreManager
from ..utils.logging import AnalyticsLogger


class AgentState(MessagesState):
    """Enhanced agent state for ReAct reasoning"""
    context: Dict[str, RunningSummary]
    session_logs: List[str]
    phone_mentions: List[str]


class OStoreAgent:
    """O!Store ReAct Agent for mobile phone retail assistance"""

    def __init__(self):
        settings.validate()

        # Initialize components
        self.vector_manager = VectorStoreManager()
        self.logger = AnalyticsLogger()

        # Get retrievers and tools
        specs_retriever, docs_retriever = self.vector_manager.get_retrievers()
        self.tools = get_retriever_tools(specs_retriever, docs_retriever)
        self.tools_by_name = {tool.name: tool for tool in self.tools}

        # Initialize models
        self.model = ChatOpenAI(
            model=settings.openai_model,
            temperature=0,
            streaming=False
        )
        self.summarization_model = self.model.bind(max_tokens=128)

        # Build workflow
        self._build_workflow()

    def _build_workflow(self):
        """Build the ReAct workflow graph"""
        # Summarization node
        summarization_node = SummarizationNode(
            token_counter=count_tokens_approximately,
            model=self.summarization_model,
            max_tokens=256,
            max_tokens_before_summary=256,
            max_summary_tokens=128,
        )

        # Build workflow
        workflow = StateGraph(AgentState)
        workflow.add_node("summarize", summarization_node)
        workflow.add_node("agent", self._call_model)
        workflow.add_node("tools", self._tool_node)
        workflow.add_node("analytics", self._analytics_node)

        workflow.set_entry_point("summarize")
        workflow.add_edge("summarize", "agent")
        workflow.add_edge("tools", "agent")
        workflow.add_edge("analytics", END)

        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "continue": "tools",
                "end": "analytics",
            }
        )

        # Compile with memory
        memory = InMemorySaver()
        self.graph = workflow.compile(checkpointer=memory)

    def _call_model(self, state: AgentState):
        """Enhanced ReAct agent with proper reasoning chain"""
        system_prompt = SystemMessage("""
Ты — виртуальный консультант O!Store. Ты помогаешь клиентам выбрать мобильные телефоны и отвечаешь на вопросы о магазине.

ИНСТРУМЕНТЫ:
- specs_retriever: для поиска характеристик и цен мобильных телефонов
- docs_retriever: для вопросов о магазине (адреса, гарантия, доставка, акции)

КЛЮЧЕВЫЕ ПРАВИЛА:
- Общайся на русском или кыргызском
- Уточняй потребности клиента (бюджет, характеристики)
- КРИТИЧНО: НЕ придумывай и НЕ галлюцинируй данные - используй только результаты инструментов
- Если инструмент не возвращает информацию, честно скажи "У меня нет доступа к этой информации"
- ВСЕГДА предлагай конкретные альтернативы для телефонов, даже если точного совпадения нет
- Структурируй ответы: бренд, модель, RAM, камеры, процессор, батарея, экран, цена
- Объясняй почему каждый телефон хороший выбор для клиента
- Предлагай 2-3 варианта для сравнения
- Завершай вопросами или предложениями дополнительной помощи

СТИЛЬ ОТВЕТОВ:
- Начинай с подходящих рекомендаций на основе критериев
- Для каждого телефона объясни: "это отличный выбор, потому что..."
- Используй структуру: "Рекомендую:", "Характеристики:", "Почему подходит:"
- Всегда заканчивай: "Хотите узнать больше о каком-то из вариантов?"

ФОРМАТИРОВАНИЕ MARKDOWN:
- Используй **жирный текст** только для названий телефонов (например: **iPhone 16 Pro**)
- Используй • или - для списков характеристик
- НЕ используй заголовки ###, просто **жирный текст** для выделения
- Разделяй параграфы пустыми строками
- Используй *курсив* умеренно для подчеркивания преимуществ
- Форматируй цены как **$799** или **799 USD**

ВАЖНО: Отвечай клиенту напрямую в формате Markdown, БЕЗ показа внутренних размышлений, форматирования типа "THOUGHT:" или "FINAL ANSWER:". Просто дай естественный дружелюбный ответ с красивым форматированием.
        """)

        # Track current query for logging
        current_message = state["messages"][-1] if state["messages"] else None
        if current_message and isinstance(current_message, HumanMessage):
            # Update session logs
            session_logs = state.get("session_logs", [])
            phone_mentions = state.get("phone_mentions", [])

            query_text = current_message.content
            session_logs.append(query_text)

            # Extract phone mentions
            new_mentions = self.logger.extract_phone_mentions(query_text)
            phone_mentions.extend(new_mentions)

            # Update state
            state["session_logs"] = session_logs
            state["phone_mentions"] = phone_mentions

        messages = [system_prompt] + state["messages"]
        response = self.model.bind_tools(self.tools).invoke(messages)

        return {
            "messages": [response],
            "session_logs": state.get("session_logs", []),
            "phone_mentions": state.get("phone_mentions", [])
        }

    def _tool_node(self, state: AgentState):
        """Execute tools based on model requests"""
        if not state["messages"]:
            return {"messages": []}

        last = state["messages"][-1]
        if not getattr(last, "tool_calls", None):
            return {"messages": []}

        outputs = []
        for call in last.tool_calls:
            result = self.tools_by_name[call["name"]].invoke(call["args"])
            from langchain_core.messages import ToolMessage
            outputs.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=call["id"],
                    name=call["name"],
                )
            )
        return {"messages": outputs}

    def _analytics_node(self, state: AgentState):
        """Log session analytics"""
        session_logs = state.get("session_logs", [])
        phone_mentions = state.get("phone_mentions", [])

        if session_logs:
            self.logger.log_session(session_logs, phone_mentions)

        return {}

    def _should_continue(self, state: AgentState):
        """Decide whether to continue with tools or end"""
        messages = state["messages"]
        last_message = messages[-1] if messages else None
        if not last_message or not getattr(last_message, "tool_calls", None):
            return "end"
        return "continue"

    def chat(self, message: str, thread_id: str = "default") -> str:
        """
        Send a message to the agent and get response

        Args:
            message: User message
            thread_id: Conversation thread identifier

        Returns:
            Agent response
        """
        config = {"configurable": {"thread_id": thread_id}}

        initial_state = {
            "messages": [HumanMessage(content=message)],
            "session_logs": [],
            "phone_mentions": []
        }

        result = self.graph.invoke(initial_state, config)

        # Extract response
        if result and "messages" in result:
            last_message = result["messages"][-1]
            if hasattr(last_message, "content") and last_message.content:
                return last_message.content

        return "Извините, произошла ошибка при обработке запроса."

    def health_check(self) -> Dict[str, Any]:
        """Check system health"""
        return {
            "vector_stores": self.vector_manager.health_check(),
            "openai_configured": bool(settings.openai_api_key),
            "timestamp": datetime.now().isoformat()
        }