import os
from typing import Annotated, TypedDict
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import Response
from twilio.twiml.voice_response import VoiceResponse
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

# --- LANGGRAPH IMPORTS ---
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# 1. Load Environment Variables
load_dotenv()

# 2. Setup Google Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# --- 3. THE RESTAURANT DATABASE ---
MENU_DB = {
    "pepperoni_small": {"name": "Small Pepperoni Pizza", "price": 10.0,
                        "tags": ["pizza", "meat", "pepperoni", "small"]},
    "pepperoni_large": {"name": "Large Pepperoni Pizza", "price": 12.0,
                        "tags": ["pizza", "meat", "pepperoni", "large"]},
    "cheese_small": {"name": "Small Cheese Pizza", "price": 8.0, "tags": ["pizza", "cheese", "small", "vegetarian"]},
    "cheese_large": {"name": "Large Cheese Pizza", "price": 10.0, "tags": ["pizza", "cheese", "large", "vegetarian"]},
    "coke": {"name": "Coke", "price": 2.0, "tags": ["drink", "soda", "coke"]},
}


# --- 4. THE CUSTOM STATE (Updated with Handoff Flag) ---
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    cart: list
    order_total: float
    requires_handoff: bool  # NEW: Tracks if the user needs a human


# --- 5. THE TOOLS (Now we have THREE tools) ---
@tool
def search_menu(query: str):
    """Use this tool to search the menu database BEFORE adding an item to the cart."""
    pass


@tool
def add_to_cart(item_id: str, quantity: int):
    """Use this tool to add an item to the cart. You MUST provide the exact item_id."""
    pass


@tool
def request_human_handoff(reason: str):
    """Use this tool IMMEDIATELY if the customer is frustrated, has a complaint, or explicitly asks to speak to a human, staff, or manager."""
    pass


# Bind ALL tools to the LLM
llm_with_tools = llm.bind_tools([search_menu, add_to_cart, request_human_handoff])

SYSTEM_PROMPT = """You are DineLine, an AI phone ordering assistant.
Rules:
1. Keep responses under 20 words.
2. You DO NOT have the menu memorized. You MUST use the `search_menu` tool.
3. When the user confirms an order, use the `add_to_cart` tool with the exact item_id.
4. NEVER calculate prices yourself.
5. If the customer asks for a human, manager, or has a complaint, YOU MUST call the `request_human_handoff` tool immediately.
6. When the user says they are done, tell them their total and say "Goodbye".
"""


# --- 6. LANGGRAPH NODES ---

def call_model(state: AgentState):
    cart_context = f"\n\nCurrent Cart: {state.get('cart', [])}\nTotal: ${state.get('order_total', 0.0)}"
    sys_msg = SystemMessage(content=SYSTEM_PROMPT + cart_context)

    messages = [sys_msg] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": response}


def execute_tools(state: AgentState):
    last_message = state["messages"][-1]
    cart = state.get("cart", [])
    order_total = state.get("order_total", 0.0)
    requires_handoff = state.get("requires_handoff", False)

    tool_messages = []

    for tool_call in last_message.tool_calls:

        # --- NEW LOGIC: Human Handoff Node ---
        if tool_call["name"] == "request_human_handoff":
            requires_handoff = True
            tool_messages.append(ToolMessage(
                content="System: Handoff initiated. End conversation gracefully.",
                tool_call_id=tool_call["id"]
            ))

        elif tool_call["name"] == "search_menu":
            query = tool_call["args"]["query"].lower()
            results = []
            for item_id, details in MENU_DB.items():
                if query in details["name"].lower() or any(query in tag for tag in details["tags"]):
                    results.append(f"ID: {item_id}, Name: {details['name']}, Price: ${details['price']}")

            if not results:
                reply = f"System: No items found matching '{query}'."
            else:
                reply = "System: Available items:\n" + "\n".join(results)

            tool_messages.append(ToolMessage(content=reply, tool_call_id=tool_call["id"]))

        elif tool_call["name"] == "add_to_cart":
            item_id = tool_call["args"]["item_id"]
            qty = tool_call["args"]["quantity"]

            item_data = MENU_DB.get(item_id)
            if not item_data:
                tool_messages.append(
                    ToolMessage(content=f"Error: Invalid item_id {item_id}", tool_call_id=tool_call["id"]))
                continue

            price = item_data["price"]
            line_total = price * qty

            cart.append({
                "item_id": item_id, "quantity": qty, "unit_price": price, "line_total": line_total
            })
            order_total += line_total

            tool_messages.append(ToolMessage(
                content=f"Successfully added {qty} {item_id}. New order total is ${order_total}",
                tool_call_id=tool_call["id"]
            ))

    # Return updated state, including handoff flag
    return {"messages": tool_messages, "cart": cart, "order_total": order_total, "requires_handoff": requires_handoff}


def router(state: AgentState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "execute_tools"
    return END


# --- 7. COMPILE THE GRAPH ---
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("execute_tools", execute_tools)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", router)
workflow.add_edge("execute_tools", "agent")

memory = MemorySaver()
app_graph = workflow.compile(checkpointer=memory)

# --- 8. FASTAPI SERVER ---
app = FastAPI()


@app.post("/voice")
async def voice_handler(request: Request):
    form_data = await request.form()
    call_sid = form_data.get("CallSid")
    user_speech = form_data.get("SpeechResult")

    config = {"configurable": {"thread_id": call_sid}}
    resp = VoiceResponse()

    if not user_speech:
        print(f"\n--- New Call Started: {call_sid} ---")

        # Initialize requires_handoff to False
        app_graph.invoke(
            {
                "messages": [SystemMessage(content=SYSTEM_PROMPT)],
                "cart": [],
                "order_total": 0.0,
                "requires_handoff": False
            },
            config=config
        )

        greeting = "Welcome to DineLine Pizza! What would you like to order today?"
        app_graph.invoke(
            {"messages": [{"role": "assistant", "content": greeting}]},
            config=config
        )

        resp.say(greeting)
        resp.gather(input="speech", action="/voice", speechTimeout="auto")
        return Response(content=str(resp), media_type="application/xml")

    print(f"\nUser said: {user_speech}")

    events = app_graph.invoke(
        {"messages": [HumanMessage(content=user_speech)]},
        config=config
    )

    ai_response = events["messages"][-1].content
    requires_handoff = events.get("requires_handoff", False)

    print(f"AI replied: {ai_response}")
    print(f"Current Cart State: {events.get('cart')}")
    print(f"Handoff Requested: {requires_handoff}")

    # --- NEW LOGIC: Intercept the conversation if handoff is flagged ---
    if requires_handoff:
        staff_number = os.getenv("STAFF_PHONE_NUMBER")
        print(f"Executing Twilio Dial to: {staff_number}")

        resp.say("Please hold while I connect you to a staff member.")

        # The <Dial> verb physically transfers the call in Twilio
        if staff_number:
            resp.dial(staff_number)
        else:
            resp.say("I'm sorry, I cannot reach a staff member at this time.")
            resp.hangup()

        return Response(content=str(resp), media_type="application/xml")

    # Existing normal behavior
    resp.say(ai_response)

    if "goodbye" in ai_response.lower():
        resp.hangup()
    else:
        resp.gather(input="speech", action="/voice", speechTimeout="auto")

    return Response(content=str(resp), media_type="application/xml")