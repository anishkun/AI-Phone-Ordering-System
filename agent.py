import os
import json
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

# Load the hierarchical JSON menu from the file
try:
    with open("menu.json", "r") as f:
        MENU_DATA = json.load(f)
        MENU_ITEMS = MENU_DATA.get("menu_items", [])
except FileNotFoundError:
    print("WARNING: menu.json not found. Make sure it is in the same directory.")
    MENU_ITEMS = []


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
def add_to_cart(variant_id: str, quantity: int):
    """Add an item to the cart. You MUST provide the exact 'variant_id' (e.g., 'pep_small')."""
    pass


@tool
def request_human_handoff(reason: str):
    """Use IMMEDIATELY if the customer is frustrated or asks for a human."""
    pass


llm_with_tools = llm.bind_tools([search_menu, add_to_cart, request_human_handoff])

SYSTEM_PROMPT = """You are DineLine, an AI phone ordering assistant.
Rules:
1. Keep responses under 20 words.
2. You MUST use the `search_menu` tool to find items, sizes, and their `variant_id`.
3. When the user confirms an order, use the `add_to_cart` tool using the exact `variant_id` of the size they chose.
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
            results = []

            # Search through the nested JSON structure
            for item in MENU_ITEMS:
                # Create a searchable string combining all relevant fields
                searchable_text = f"{item['name']} {item['description']} {' '.join(item.get('tags', []))} {' '.join(item.get('allergens', []))}".lower()

                if query in searchable_text:
                    # Format the variants so the LLM knows the prices and exact variant_ids to use
                    variants_str = " | ".join(
                        [f"{v['name']} (ID: {v['variant_id']}, ${v['price']})" for v in item['variants']])
                    veg = "Vegetarian" if item.get("is_vegetarian") else "Not Vegetarian"
                    spice = item.get("spice_level", "none")

                    results.append(
                        f"- {item['name']} ({veg}, Spice: {spice}): {item['description']} -> Options: [{variants_str}]")

            reply = "System: Available items:\n" + "\n".join(
                results) if results else f"System: No items found matching '{query}'."
            tool_messages.append(ToolMessage(content=reply, tool_call_id=tool_call["id"]))

        elif tool_call["name"] == "add_to_cart":
            variant_id = tool_call["args"]["variant_id"]
            qty = tool_call["args"]["quantity"]

            # Deep search to find the specific variant across all menu items
            found_variant = None
            found_item_name = None

            for item in MENU_ITEMS:
                for var in item["variants"]:
                    if var["variant_id"] == variant_id:
                        found_variant = var
                        found_item_name = item["name"]
                        break
                if found_variant:
                    break

            if not found_variant:
                tool_messages.append(
                    ToolMessage(
                        content=f"Error: Invalid variant_id '{variant_id}'. You must search the menu first to find the correct ID.",
                        tool_call_id=tool_call["id"]))
                continue

            # Calculate and structure the POS payload
            line_total = found_variant["price"] * qty
            cart.append({
                "variant_id": variant_id,
                "name": f"{found_item_name} - {found_variant['name']}",
                "quantity": qty,
                "unit_price": found_variant["price"],
                "line_total": line_total
            })
            order_total += line_total
            tool_messages.append(
                ToolMessage(content=f"Added {qty} {found_item_name} ({found_variant['name']}). Total: ${order_total}",
                            tool_call_id=tool_call["id"]))

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