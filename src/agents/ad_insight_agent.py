"""
Ad Performance Insight Agent — LangGraph-based multi-step reasoning pipeline.

Architecture:
  User Query → Router Agent → [SQL Tool | RAG Tool | Calculator Tool]
             → Synthesizer → Insight Response

Latency target: sub-200 ms p95 across 10,000+ queries.
"""

import time
import logging
from typing import TypedDict, Annotated, Sequence, Optional
from functools import wraps

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_aws import ChatBedrock
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import operator

from src.agents.sql_tool import execute_ad_sql_query
from src.rag.vector_store import retrieve_ad_context
from src.utils.latency_tracker import LatencyTracker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

tracker = LatencyTracker()


# ── State ─────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    query_id: str
    latency_ms: float
    tool_calls_made: int
    insights_generated: int


# ── Tools ─────────────────────────────────────────────────────────────────────

@tool
def sql_query_tool(sql: str) -> str:
    """Execute a SQL query against the advertising dataset.
    Use this to retrieve structured metrics like CTR, ROAS, CPC, impressions,
    conversions, spend, and revenue grouped by campaign, ad group, or date.
    """
    return execute_ad_sql_query(sql)


@tool
def rag_context_tool(question: str) -> str:
    """Retrieve semantic context from the ad performance knowledge base.
    Use this to answer qualitative questions about ad strategy, industry
    benchmarks, campaign best practices, or historical trend narratives.
    """
    return retrieve_ad_context(question)


@tool
def calculate_metric_tool(expression: str) -> str:
    """Safely evaluate a mathematical expression for metric calculations.
    Supports: +, -, *, /, **, %, and common functions (round, abs, min, max).
    Example: 'round((clicks / impressions) * 100, 2)'
    """
    import ast
    import math

    allowed_names = {
        "round": round, "abs": abs, "min": min, "max": max,
        "sum": sum, "pow": pow, "sqrt": math.sqrt, "log": math.log,
    }
    try:
        tree = ast.parse(expression, mode="eval")
        # Safety: only allow safe node types
        for node in ast.walk(tree):
            if not isinstance(node, (
                ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call,
                ast.Num, ast.Constant, ast.Name, ast.Load,
                ast.Add, ast.Sub, ast.Mul, ast.Div, ast.Pow, ast.Mod,
                ast.USub, ast.UAdd,
            )):
                return f"Error: Unsafe expression node '{type(node).__name__}'"
        result = eval(compile(tree, "<string>", "eval"), {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as e:
        return f"Calculation error: {e}"


TOOLS = [sql_query_tool, rag_context_tool, calculate_metric_tool]


# ── LLM Factory ───────────────────────────────────────────────────────────────

def build_llm(backend: str = "bedrock", model_id: Optional[str] = None):
    """Instantiate the LLM backend. Supports AWS Bedrock and OpenAI."""
    if backend == "bedrock":
        return ChatBedrock(
            model_id=model_id or "anthropic.claude-3-sonnet-20240229-v1:0",
            model_kwargs={"temperature": 0, "max_tokens": 2048},
        )
    elif backend == "openai":
        return ChatOpenAI(
            model=model_id or "gpt-4o",
            temperature=0,
            max_tokens=2048,
        )
    else:
        raise ValueError(f"Unsupported backend: {backend}")


# ── Graph Nodes ───────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert advertising analyst. Your job is to answer
questions about ad campaign performance by reasoning step-by-step.

Available tools:
- sql_query_tool: query structured advertising data (impressions, clicks, spend, revenue, CTR, ROAS, CPC)
- rag_context_tool: retrieve qualitative context, benchmarks, or strategic insights
- calculate_metric_tool: perform arithmetic on retrieved numbers

Always:
1. Start by identifying what data you need.
2. Use sql_query_tool for quantitative lookups.
3. Use rag_context_tool for qualitative context or benchmarks.
4. Synthesize a clear, data-driven insight with concrete numbers and recommendations.
5. Be concise — aim for actionable answers, not essays.
"""


def build_agent_node(llm_with_tools):
    def agent_node(state: AgentState) -> AgentState:
        messages = state["messages"]
        if not any(hasattr(m, "role") and m.role == "system" for m in messages):
            from langchain_core.messages import SystemMessage
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)

        start = time.perf_counter()
        response = llm_with_tools.invoke(messages)
        elapsed = (time.perf_counter() - start) * 1000

        tracker.record(elapsed)
        logger.info(f"[Agent] LLM call: {elapsed:.1f} ms | tool_calls={len(getattr(response, 'tool_calls', []))}")

        return {
            "messages": [response],
            "latency_ms": state.get("latency_ms", 0) + elapsed,
            "tool_calls_made": state.get("tool_calls_made", 0) + len(getattr(response, "tool_calls", [])),
        }
    return agent_node


def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


# ── Graph Assembly ─────────────────────────────────────────────────────────────

def build_graph(backend: str = "bedrock", model_id: Optional[str] = None) -> StateGraph:
    llm = build_llm(backend, model_id)
    llm_with_tools = llm.bind_tools(TOOLS)

    graph = StateGraph(AgentState)
    graph.add_node("agent", build_agent_node(llm_with_tools))
    graph.add_node("tools", ToolNode(TOOLS))

    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile()


# ── Public Interface ──────────────────────────────────────────────────────────

_graph_cache: dict = {}


def get_compiled_graph(backend: str = "bedrock", model_id: Optional[str] = None):
    key = f"{backend}:{model_id}"
    if key not in _graph_cache:
        _graph_cache[key] = build_graph(backend, model_id)
    return _graph_cache[key]


def run_insight_query(
    query: str,
    query_id: str = "q-0",
    backend: str = "bedrock",
    model_id: Optional[str] = None,
) -> dict:
    """
    Run a single ad-insight query through the agent graph.

    Returns:
        {
          "query_id": str,
          "answer": str,
          "latency_ms": float,
          "tool_calls_made": int,
        }
    """
    graph = get_compiled_graph(backend, model_id)
    initial_state: AgentState = {
        "messages": [HumanMessage(content=query)],
        "query_id": query_id,
        "latency_ms": 0.0,
        "tool_calls_made": 0,
        "insights_generated": 0,
    }

    wall_start = time.perf_counter()
    final_state = graph.invoke(initial_state)
    wall_ms = (time.perf_counter() - wall_start) * 1000

    answer = ""
    for msg in reversed(final_state["messages"]):
        if isinstance(msg, AIMessage) and msg.content:
            answer = msg.content
            break

    return {
        "query_id": query_id,
        "answer": answer,
        "latency_ms": round(wall_ms, 2),
        "tool_calls_made": final_state.get("tool_calls_made", 0),
    }
