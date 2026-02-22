import os
import json
import base64
import asyncio
import httpx
import websockets
from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from twilio.twiml.voice_response import VoiceResponse
from twilio.rest import Client
from langchain_core.messages import HumanMessage, SystemMessage

# --- IMPORT THE BRAIN FROM AGENT.PY ---
from agent import app_graph, SYSTEM_PROMPT

load_dotenv()


# --- 1. EXTERNAL API HELPERS (The Mouth) ---
async def generate_tts(text: str) -> str:
    # FIX APPLIED: Guard clause to prevent crashing on empty tool outputs
    if not text or not str(text).strip():
        return ""

    """Takes text, calls ElevenLabs, returns Base64 encoded mu-law 8000Hz audio for Twilio."""
    voice_id = os.getenv("ELEVENLABS_VOICE_ID", "pNInz6obpgDQGcFmaJcg")
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}?output_format=ulaw_8000"
    headers = {"xi-api-key": os.getenv("ELEVENLABS_API_KEY")}
    payload = {"text": str(text), "model_id": "eleven_turbo_v2_5"}

    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()  # Fail safely if ElevenLabs is down
        return base64.b64encode(response.content).decode('utf-8')


# --- 2. FASTAPI & WEBSOCKET SERVER ---
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

    # Connect to Deepgram (The Ears) with endpointing set to 500ms
    deepgram_url = "wss://api.deepgram.com/v1/listen?encoding=mulaw&sample_rate=8000&channels=1&endpointing=500"

    try:
        async with websockets.connect(
                deepgram_url,
                extra_headers={"Authorization": f"Token {os.getenv('DEEPGRAM_API_KEY')}"}
        ) as deepgram_ws:

            call_sid = None
            stream_sid = None
            config = None

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

                            await app_graph.ainvoke(
                                {"messages": [SystemMessage(content=SYSTEM_PROMPT)], "cart": [], "order_total": 0.0,
                                 "requires_handoff": False}, config=config)
                            greeting = "Welcome to DineLine Pizza! What would you like to order today?"
                            await app_graph.ainvoke({"messages": [{"role": "assistant", "content": greeting}]},
                                                    config=config)

                            print(f"AI: {greeting}")
                            audio_payload = await generate_tts(greeting)
                            if audio_payload:
                                await websocket.send_text(json.dumps(
                                    {"event": "media", "streamSid": stream_sid, "media": {"payload": audio_payload}}))

                        elif message["event"] == "media":
                            audio_bytes = base64.b64decode(message["media"]["payload"])
                            await deepgram_ws.send(audio_bytes)

                        elif message["event"] == "stop":
                            # FIX APPLIED: Close deepgram to prevent memory leaks
                            await deepgram_ws.close()
                            break
                except Exception as e:
                    print(f"Twilio Listen Error: {e}")

            async def listen_to_deepgram():
                try:
                    while True:
                        dg_response = await deepgram_ws.recv()
                        result = json.loads(dg_response)

                        if result.get("is_final") and result.get("speech_final") and \
                                result.get("channel", {}).get("alternatives", [{}])[0].get("transcript"):
                            user_speech = result["channel"]["alternatives"][0]["transcript"]
                            print(f"User said: {user_speech}")

                            events = await app_graph.ainvoke({"messages": [HumanMessage(content=user_speech)]},
                                                             config=config)

                            # Parse out list responses if Gemini returns tool blocks
                            raw_content = events["messages"][-1].content
                            if isinstance(raw_content, list):
                                ai_response = " ".join(
                                    [block["text"] for block in raw_content if block.get("type") == "text"])
                            else:
                                ai_response = raw_content

                            requires_handoff = events.get("requires_handoff", False)

                            print(f"AI replied: {ai_response}")

                            if requires_handoff:
                                staff_number = os.getenv("STAFF_PHONE_NUMBER")
                                print(f"Handoff Triggered. Bridging call to: {staff_number}")
                                twiml_handoff = f'<Response><Say>Connecting you to staff.</Say><Dial>{staff_number}</Dial></Response>'

                                await asyncio.to_thread(twilio_client.calls(call_sid).update, twiml=twiml_handoff)
                                break

                            audio_payload = await generate_tts(ai_response)
                            if audio_payload:
                                await websocket.send_text(json.dumps(
                                    {"event": "media", "streamSid": stream_sid, "media": {"payload": audio_payload}}))

                except Exception as e:
                    print(f"Deepgram Listen Error: {e}")

            await asyncio.gather(listen_to_twilio(), listen_to_deepgram())

    except WebSocketDisconnect:
        print("WebSocket disconnected naturally.")
    except Exception as e:
        print(f"WebSocket Error: {e}")