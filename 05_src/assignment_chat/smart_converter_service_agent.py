import os
from typing import Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from __init__ import API_GATEWAY_URL

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".secrets"))

# ── Same conversion table as smart_converter_service.py ──────────────────────

_CONVERSIONS = {
    ("celsius", "fahrenheit"):   lambda x: x * 9 / 5 + 32,
    ("fahrenheit", "celsius"):   lambda x: (x - 32) * 5 / 9,
    ("celsius", "kelvin"):       lambda x: x + 273.15,
    ("kelvin", "celsius"):       lambda x: x - 273.15,
    ("fahrenheit", "kelvin"):    lambda x: (x - 32) * 5 / 9 + 273.15,
    ("kelvin", "fahrenheit"):    lambda x: (x - 273.15) * 9 / 5 + 32,
    ("kilometers", "miles"):     lambda x: x * 0.621371,
    ("miles", "kilometers"):     lambda x: x * 1.60934,
    ("meters", "feet"):          lambda x: x * 3.28084,
    ("feet", "meters"):          lambda x: x / 3.28084,
    ("kilograms", "pounds"):     lambda x: x * 2.20462,
    ("pounds", "kilograms"):     lambda x: x / 2.20462,
}


# ── Tool (the LLM decides when to call this) ──────────────────────────────────

@tool
def convert_units(value: float, from_unit: str, to_unit: str) -> str:
    """
    Convert a numeric value from one unit to another.
    Supported categories:
    - Temperature: celsius, fahrenheit, kelvin
    - Length: kilometers, miles, meters, feet
    - Weight: kilograms, pounds
    Use this whenever the user asks to convert a measurement.
    """
    key = (from_unit.lower(), to_unit.lower())
    if key in _CONVERSIONS:
        result = round(_CONVERSIONS[key](value), 4)
        return f"{value} {from_unit} = {result} {to_unit}"
    return (
        f"Conversion from '{from_unit}' to '{to_unit}' is not supported. "
        "Supported units: celsius, fahrenheit, kelvin, kilometers, miles, "
        "meters, feet, kilograms, pounds."
    )


# ── LangGraph agent setup (built once at import time) ─────────────────────────

_tools = [convert_units]

_llm = ChatOpenAI(
    base_url=API_GATEWAY_URL,
    model="gpt-4o-mini",
    temperature=0,
    api_key="any value",
    default_headers={"x-api-key": os.getenv("API_GATEWAY_KEY")},
).bind_tools(_tools)


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def _agent_node(state: AgentState) -> dict:
    """Run the LLM. It either calls a tool or returns a final answer."""
    response = _llm.invoke(state["messages"])
    return {"messages": [response]}


def _should_use_tools(state: AgentState) -> str:
    """Router: check whether the LLM's last message contains a tool call."""
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


_builder = StateGraph(AgentState)
_builder.add_node("agent", _agent_node)
_builder.add_node("tools", ToolNode(_tools))
_builder.add_edge(START, "agent")
_builder.add_conditional_edges(
    "agent",
    _should_use_tools,
    {"tools": "tools", END: END},
)
_builder.add_edge("tools", "agent")  # loop back after every tool call

_graph = _builder.compile()

_SYSTEM_MSG = SystemMessage(content=(
    "You are a unit conversion assistant. "
    "Always use the convert_units function to answer conversion questions. "
    "Supported categories: temperature (Celsius, Fahrenheit, Kelvin), "
    "length (kilometers, miles, meters, feet), "
    "weight (kilograms, pounds)."
))


# ── Public interface — identical signature to smart_converter_service.py ──────
#
#  Flow for a conversion query (e.g. "100 Fahrenheit to Celsius"):
#
#    START
#      │
#      ▼
#    [agent]  ← LLM sees SystemMessage + HumanMessage
#      │         decides to call convert_units(100, "fahrenheit", "celsius")
#      ▼
#    [tools]  ← executes convert_units → "100 fahrenheit = 37.7778 celsius"
#      │         result added to messages as ToolMessage
#      ▼
#    [agent]  ← LLM sees all 4 messages, writes natural-language answer
#      │         no further tool calls
#      ▼
#     END
#
#  Flow for a general question the LLM can answer directly (no tool needed):
#
#    START → [agent] → no tool calls → END

def smart_converter(query: str) -> str:
    if not query.strip():
        return (
            "Please enter a conversion request, "
            "e.g. *'100 Fahrenheit to Celsius'* or *'5 miles to kilometers'*."
        )
    try:
        final_state = _graph.invoke(
            {"messages": [_SYSTEM_MSG, HumanMessage(content=query)]}
        )
        answer = final_state["messages"][-1].content
        return f"I delegated this to my converter agent. Here's its reply:\n\n{answer}"
    except Exception as e:
        return f"Error: {str(e)}"
