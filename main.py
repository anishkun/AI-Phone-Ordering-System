import os
import json
from typing import Annotated, TypedDict
from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from twilio.twiml.voice_response import VoiceResponse
from twilio.rest import Client  # Required to modify live calls during handoff
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


# --- 4. THE CUSTOM STATE ---
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    cart: list
    order_total: float
    requires_handoff: bool


# --- 5. THE TOOLS ---
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

# --- 8. FASTAPI SERVER (NOW USING WEBSOCKETS) ---
app = FastAPI()

# Required to modify live calls during a stream (for Human Handoff)
twilio_client = Client(os.getenv("TWILIO_ACCOUNT_SID"), os.getenv("TWILIO_AUTH_TOKEN"))


@app.post("/voice")
async def voice_handler(request: Request):
    """
    Step 1: When the phone rings, we do NOT use <Gather>.
    We immediately open a WebSocket <Stream>.
    """
    resp = VoiceResponse()

    # Convert ngrok https:// URL to wss:// for WebSockets
    host = request.url.hostname
    wss_url = f"wss://{host}/stream"

    print(f"Incoming call. Connecting to Media Stream at: {wss_url}")

    # Start the bidirectional audio stream
    connect = resp.connect()
    connect.stream(url=wss_url)

    return Response(content=str(resp), media_type="application/xml")


@app.websocket("/stream")
async def websocket_endpoint(websocket: WebSocket):
    """
    Step 2: Twilio streams the raw audio bytes here every 20ms.
    """
    await websocket.accept()
    call_sid = None
    stream_sid = None
    config = None

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            # --- EVENT: STREAM STARTED ---
            if message["event"] == "start":
                stream_sid = message["start"]["streamSid"]
                call_sid = message["start"]["callSid"]
                config = {"configurable": {"thread_id": call_sid}}

                print(f"\n--- [WebSocket] Call Started: {call_sid} ---")

                # Initialize LangGraph State
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
                app_graph.invoke({"messages": [{"role": "assistant", "content": greeting}]}, config=config)

                print(f"AI: {greeting}")

                # -----------------------------------------------------------------
                # REQUIRED INTEGRATION:
                # You must run `greeting` through a Text-to-Speech API here,
                # encode the audio to base64 mu-law, and send it to Twilio via:
                # await websocket.send_text(json.dumps({"event": "media", "streamSid": stream_sid, "media": {"payload": audio_payload}}))
                # -----------------------------------------------------------------

            # --- EVENT: INCOMING AUDIO FROM CALLER ---
            elif message["event"] == "media":
                # The raw audio bytes from the caller (Base64 encoded, 8000Hz, mu-law)
                inbound_audio_base64 = message["media"]["payload"]

                # -----------------------------------------------------------------
                # REQUIRED INTEGRATION:
                # Pipe `inbound_audio_base64` into a real-time Speech-to-Text API
                # (like Deepgram) here.
                # -----------------------------------------------------------------

                # ---> PSEUDO-CODE: When your STT API detects the user stopped talking:
                user_is_done_speaking = False
                user_speech = ""  # The text returned from your STT API

                if user_is_done_speaking:
                    print(f"User said: {user_speech}")

                    # Process through LangGraph
                    events = app_graph.invoke(
                        {"messages": [HumanMessage(content=user_speech)]},
                        config=config
                    )

                    ai_response = events["messages"][-1].content
                    requires_handoff = events.get("requires_handoff", False)

                    print(f"AI replied: {ai_response}")

                    # Handle Handoff Request over WebSocket
                    if requires_handoff:
                        staff_number = os.getenv("STAFF_PHONE_NUMBER")
                        print(f"Handoff Triggered. Bridging call to: {staff_number}")

                        # Modify the live call using Twilio REST API to stop the stream and dial the staff
                        twiml_handoff = f'<Response><Say>Connecting you to staff.</Say><Dial>{staff_number}</Dial></Response>'
                        twilio_client.calls(call_sid).update(twiml=twiml_handoff)
                        break  # End websocket connection gracefully

                    # -----------------------------------------------------------------
                    # REQUIRED INTEGRATION:
                    # If no handoff, run `ai_response` through your Text-to-Speech API
                    # and send the audio bytes back down the websocket.
                    # -----------------------------------------------------------------

            # --- EVENT: STREAM STOPPED ---
            elif message["event"] == "stop":
                print(f"--- [WebSocket] Stream Ended: {call_sid} ---")
                break

    except WebSocketDisconnect:
        print("WebSocket disconnected naturally.")
    except Exception as e:
        print(f"WebSocket Error: {e}")