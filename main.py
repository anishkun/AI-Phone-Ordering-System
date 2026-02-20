import os
import json
import base64
import asyncio
import httpx
import websockets
from typing import Annotated, TypedDict
from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from twilio.twiml.voice_response import VoiceResponse
from twilio.rest import Client
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

# --- 1. SETUP & DATABASES ---
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",  # <--- CHANGE THIS FROM 1.5 TO 2.5
    temperature=0.0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

MENU_DB = {
    "pepperoni_small": {"name": "Small Pepperoni Pizza", "price": 10.0,
                        "tags": ["pizza", "meat", "pepperoni", "small"]},
    "pepperoni_large": {"name": "Large Pepperoni Pizza", "price": 12.0,
                        "tags": ["pizza", "meat", "pepperoni", "large"]},
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
    cart = state.get("cart", [])
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
app_graph = workflow.compile(checkpointer=MemorySaver())


# --- 3. EXTERNAL API HELPERS (The Mouth) ---
async def generate_tts(text: str) -> str:
    """Takes text, calls ElevenLabs, returns Base64 encoded mu-law 8000Hz audio for Twilio."""
    voice_id = os.getenv("ELEVENLABS_VOICE_ID", "pNInz6obpgDQGcFmaJcg")
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}?output_format=ulaw_8000"
    headers = {"xi-api-key": os.getenv("ELEVENLABS_API_KEY")}
    payload = {"text": text, "model_id": "eleven_turbo_v2_5"}

    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload, headers=headers)
        return base64.b64encode(response.content).decode('utf-8')


# --- 4. FASTAPI & WEBSOCKET SERVER ---
app = FastAPI()
twilio_client = Client(os.getenv("TWILIO_ACCOUNT_SID"), os.getenv("TWILIO_AUTH_TOKEN"))


@app.post("/voice")
async def voice_handler(request: Request):
    resp = VoiceResponse()
    host = request.url.hostname
    wss_url = f"wss://{host}/stream"

    print(f"Incoming call. Connecting to Media Stream at: {wss_url}")
    resp.connect().stream(url=wss_url)
    return Response(content=str(resp), media_type="application/xml")


@app.websocket("/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    # Connect to Deepgram (The Ears) with endpointing set to 500ms for fast replies
    deepgram_url = "wss://api.deepgram.com/v1/listen?encoding=mulaw&sample_rate=8000&channels=1&endpointing=500"

    try:
        async with websockets.connect(
                deepgram_url,
                extra_headers={"Authorization": f"Token {os.getenv('DEEPGRAM_API_KEY')}"}
        ) as deepgram_ws:

            call_sid = None
            stream_sid = None
            config = None

            # Task 1: Receive audio from Twilio and forward to Deepgram
            async def listen_to_twilio():
                nonlocal call_sid, stream_sid, config
                try:
                    while True:
                        data = await websocket.receive_text()
                        message = json.loads(data)

                        if message["event"] == "start":
                            stream_sid = message["start"]["streamSid"]
                            call_sid = message["start"]["callSid"]
                            config = {"configurable": {"thread_id": call_sid}}

                            # FIX 1 APPLIED HERE: Initialize LangGraph & Greeting asynchronously
                            await app_graph.ainvoke(
                                {"messages": [SystemMessage(content=SYSTEM_PROMPT)], "cart": [], "order_total": 0.0,
                                 "requires_handoff": False}, config=config)
                            greeting = "Welcome to DineLine Pizza! What would you like to order today?"
                            await app_graph.ainvoke({"messages": [{"role": "assistant", "content": greeting}]},
                                                    config=config)

                            print(f"AI: {greeting}")
                            audio_payload = await generate_tts(greeting)
                            await websocket.send_text(json.dumps(
                                {"event": "media", "streamSid": stream_sid, "media": {"payload": audio_payload}}))

                        elif message["event"] == "media":
                            # Decode Twilio base64 audio and send raw bytes to Deepgram
                            audio_bytes = base64.b64decode(message["media"]["payload"])
                            await deepgram_ws.send(audio_bytes)

                        elif message["event"] == "stop":
                            break
                except Exception as e:
                    print(f"Twilio Listen Error: {e}")

            # Task 2: Receive transcripts from Deepgram and trigger LangGraph
            async def listen_to_deepgram():
                try:
                    while True:
                        dg_response = await deepgram_ws.recv()
                        result = json.loads(dg_response)

                        # FIX 2 APPLIED HERE: Added speech_final to ensure user finished speaking
                        if result.get("is_final") and result.get("speech_final") and \
                                result.get("channel", {}).get("alternatives", [{}])[0].get("transcript"):
                            user_speech = result["channel"]["alternatives"][0]["transcript"]
                            print(f"User said: {user_speech}")

                            # FIX 1 APPLIED HERE: Use await and ainvoke
                            events = await app_graph.ainvoke({"messages": [HumanMessage(content=user_speech)]},
                                                             config=config)
                            ai_response = events["messages"][-1].content
                            requires_handoff = events.get("requires_handoff", False)

                            print(f"AI replied: {ai_response}")

                            # Handle Handoff Request
                            if requires_handoff:
                                staff_number = os.getenv("STAFF_PHONE_NUMBER")
                                print(f"Handoff Triggered. Bridging call to: {staff_number}")
                                twiml_handoff = f'<Response><Say>Connecting you to staff.</Say><Dial>{staff_number}</Dial></Response>'

                                # FIX 3 APPLIED HERE: Push Twilio HTTP request to a background thread
                                await asyncio.to_thread(twilio_client.calls(call_sid).update, twiml=twiml_handoff)
                                break

                            # Convert AI text to Speech and send to Twilio
                            audio_payload = await generate_tts(ai_response)
                            await websocket.send_text(json.dumps(
                                {"event": "media", "streamSid": stream_sid, "media": {"payload": audio_payload}}))

                except Exception as e:
                    print(f"Deepgram Listen Error: {e}")

            # Run both tasks concurrently
            await asyncio.gather(listen_to_twilio(), listen_to_deepgram())

    except WebSocketDisconnect:
        print("WebSocket disconnected naturally.")
    except Exception as e:
        print(f"WebSocket Error: {e}")