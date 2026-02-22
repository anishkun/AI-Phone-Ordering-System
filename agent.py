import os
from typing import Annotated, TypedDict
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.tools import tool

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

# --- 1. SETUP & DATABASES ---
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

MENU_DB = {
    "pepperoni_small": {"name": "Small Pepperoni Pizza", "price": 10.0, "tags": ["pizza", "meat", "pepperoni", "small"]},
    "pepperoni_large": {"name": "Large Pepperoni Pizza", "price": 12.0, "tags": ["pizza", "meat", "pepperoni", "large"]},
    "cheese_small": {"name": "Small Cheese Pizza", "price": 8.0, "tags": ["pizza", "cheese", "small", "vegetarian"]},
    "cheese_large": {"name": "Large Cheese Pizza", "price": 10.0, "tags": ["pizza", "cheese", "large", "vegetarian"]},
    "coke": {"name": "Coke", "price": 2.0, "tags": ["drink", "soda", "coke"]},
}

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    cart: list
    order_total: float
    requires_handoff: bool

# --- 2. THE TOOLS & GRAPH ---
@tool
def search_menu(query: str):
    """Search the menu database BEFORE adding an item to the cart."""
    pass

@tool
def add_to_cart(item_id: str, quantity: int):
    """Add an item to the cart. You MUST provide the exact item_id."""
    pass

@tool
def request_human_handoff(reason: str):
    """Use IMMEDIATELY if the customer is frustrated or asks for a human."""
    pass

llm_with_tools = llm.bind_tools([search_menu, add_to_cart, request_human_handoff])

SYSTEM_PROMPT = """You are DineLine, an AI phone ordering assistant.
Rules:
1. Keep responses under 20 words.
2. You MUST use the `search_menu` tool.
3. When the user confirms an order, use the `add_to_cart` tool.
4. NEVER calculate prices yourself.
5. If the customer asks for a human, call the `request_human_handoff` tool immediately.
6. DO NOT say "Goodbye" until the user explicitly says they are finished ordering (e.g., "that's all", "I'm done").
7. When they are finally done, tell them their total and say "Goodbye".
"""

def call_model(state: AgentState):
    cart_context = f"\n\nCurrent Cart: {state.get('cart', [])}\nTotal: ${state.get('order_total', 0.0)}"
    sys_msg = SystemMessage(content=SYSTEM_PROMPT + cart_context)
    response = llm_with_tools.invoke([sys_msg] + state["messages"])
    return {"messages": response}

def execute_tools(state: AgentState):
    last_message = state["messages"][-1]
    # FIX APPLIED: Cast list to prevent state mutation bugs
    cart = list(state.get("cart", []))
    order_total = state.get("order_total", 0.0)
    requires_handoff = state.get("requires_handoff", False)
    tool_messages = []

    for tool_call in last_message.tool_calls:
        if tool_call["name"] == "request_human_handoff":
            requires_handoff = True
            tool_messages.append(ToolMessage(content="System: Handoff initiated.", tool_call_id=tool_call["id"]))

        elif tool_call["name"] == "search_menu":
            query = tool_call["args"]["query"].lower()
            results = [f"ID: {i}, Name: {d['name']}, Price: ${d['price']}" for i, d in MENU_DB.items() if
                       query in d["name"].lower() or any(query in t for t in d["tags"])]
            reply = "System: Available items:\n" + "\n".join(
                results) if results else f"System: No items found matching '{query}'."
            tool_messages.append(ToolMessage(content=reply, tool_call_id=tool_call["id"]))

        elif tool_call["name"] == "add_to_cart":
            item_id = tool_call["args"]["item_id"]
            qty = tool_call["args"]["quantity"]
            item_data = MENU_DB.get(item_id)
            if not item_data:
                tool_messages.append(
                    ToolMessage(content=f"Error: Invalid item_id {item_id}", tool_call_id=tool_call["id"]))
                continue

            line_total = item_data["price"] * qty
            cart.append(
                {"item_id": item_id, "quantity": qty, "unit_price": item_data["price"], "line_total": line_total})
            order_total += line_total
            tool_messages.append(
                ToolMessage(content=f"Added {qty} {item_id}. Total: ${order_total}", tool_call_id=tool_call["id"]))

    return {"messages": tool_messages, "cart": cart, "order_total": order_total, "requires_handoff": requires_handoff}

def router(state: AgentState):
    if state["messages"][-1].tool_calls:
        return "execute_tools"
    return END

workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("execute_tools", execute_tools)
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", router)
workflow.add_edge("execute_tools", "agent")

# EXPORT THIS FOR MAIN.PY TO USE
app_graph = workflow.compile(checkpointer=MemorySaver())