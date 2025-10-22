"""
O!Store ReAct Agent - Main agent implementation
"""
import os
os.environ["HTTP_PROXY"] = "http://172.27.129.0:3128"
os.environ["HTTPS_PROXY"] = "http://172.27.129.0:3128"
from ..utils.logging import AnalyticsLogger
from ..core.vectorstore import VectorStoreManager
from ..core.config import settings
from .tools import get_retriever_tools
from langmem.short_term import SummarizationNode, RunningSummary
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, MessagesState, END
from langchain_openai import ChatOpenAI
from langchain_core.messages.utils import count_tokens_approximately
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import SystemMessage, HumanMessage
from typing import Dict, Any, List
from datetime import datetime

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
        # Initialize conversation state storage per thread
        self.state_per_thread = {}

        # Keep memory object alive for state graph
        self.memory = InMemorySaver()

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
        # Compile with persistent memory
        self.graph = workflow.compile(checkpointer=self.memory)

    def _call_model(self, state: AgentState):
        """Enhanced ReAct agent with proper reasoning chain"""
        system_prompt = SystemMessage("""
You are the virtual consultant for O!Store, helping customers select mobile phones and answering their questions about the store.
TOOLS:  
- specs_retriever: retrieves mobile phone specifications and prices  
- docs_retriever: provides information about the store (locations, warranty, delivery, promotions)  
KEY RULES:  
- Communicate in the language the user chooses or asks in  
- Clarify the customer’s needs (budget, preferred features)  
- CRUCIAL: Never invent or hallucinate data — rely exclusively on the information from the tools  
- If a tool returns no information, respond honestly: "I don’t have access to that information"  
- Always suggest specific alternative phone options, even when no exact match is available  
- Structure responses clearly with: brand, model, RAM, cameras, processor, battery, display, price  
- Explain why each phone is a good fit for the customer’s needs  
- Provide 2–3 options for comparison  
- End with questions or offers of additional assistance  
RESPONSE STYLE:  
- Begin with tailored recommendations based on the customer’s criteria  
- For each phone, explain: “This is an excellent choice because...”  
- Use this structure: “Recommend:”, “Specifications:”, “Why it fits:”  
- Always conclude with: “Would you like to know more about any of these options?”  
MARKDOWN FORMATTING:  
- Bold phone models only (e.g., **iPhone 16 Pro**)  
- Use • or - for listing features  
- Do NOT use headings like ###; use bold text for emphasis instead  
- Separate paragraphs with blank lines  
- Use *italics* sparingly to highlight advantages  
- Format prices as **$799** or **799 USD**  
IMPORTANT:  
Respond directly to the customer in Markdown format, WITHOUT revealing internal thoughts or notes such as "THOUGHT:" or "FINAL ANSWER:". Provide clear, natural, and friendly answers with neat, attractive formatting.
        """)
        # Track current query for logging
        current_message = state["messages"][-1] if state["messages"] else None
        if current_message and isinstance(current_message, HumanMessage):
            session_logs = state.get("session_logs", [])
            phone_mentions = state.get("phone_mentions", [])
            query_text = current_message.content
            session_logs.append(query_text)
            new_mentions = self.logger.extract_phone_mentions(query_text)
            phone_mentions.extend(new_mentions)
            state["session_logs"] = session_logs
            state["phone_mentions"] = phone_mentions

        # Ensure the full conversation history (user + assistant messages) is sent every time
        messages = [system_prompt] + state["messages"]

        response = self.model.bind_tools(self.tools).invoke(messages)

        # Return response wrapped in a list, preserving other state fields
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
        # Load conversation state or create new one
        state = self.state_per_thread.get(thread_id, {
            "messages": [],
            "session_logs": [],
            "phone_mentions": []
        })

        # Append new user message
        state["messages"].append(HumanMessage(content=message))

        # Debug: print message counts
        print(f"[{thread_id}] Before invoke, messages count: {len(state['messages'])}")
        for m in state["messages"]:
            prefix = "User" if isinstance(m, HumanMessage) else "Agent"
            print(f" - {prefix}: {m.content[:80]}")

        config = {"configurable": {"thread_id": thread_id}}

        # Invoke graph with current state
        result = self.graph.invoke(state, config)

        # Append all *non-user* messages returned (usually model replies)
        for msg in result.get("messages", []):
            if not isinstance(msg, HumanMessage):
                state["messages"].append(msg)

        # Update session logs and phone mentions
        state["session_logs"] = result.get("session_logs", state.get("session_logs", []))
        state["phone_mentions"] = result.get("phone_mentions", state.get("phone_mentions", []))

        # Save updated conversation state back
        self.state_per_thread[thread_id] = state

        # Debug: print updated message count
        print(f"[{thread_id}] After invoke, messages count: {len(state['messages'])}")

        # Return last assistant message content if available
        for msg in reversed(result.get("messages", [])):
            if not isinstance(msg, HumanMessage) and hasattr(msg, "content") and msg.content:
                return msg.content

        return "Извините, произошла ошибка при обработке запроса."

    def health_check(self) -> Dict[str, Any]:
        """Check system health"""
        return {
            "vector_stores": self.vector_manager.health_check(),
            "openai_configured": bool(settings.openai_api_key),
            "timestamp": datetime.now().isoformat()
        }