import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Form
from fastapi.responses import Response
from twilio.twiml.voice_response import VoiceResponse
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# 1. Load Environment Variables (Your API Key)
load_dotenv()

# 2. Setup Google Gemini
# We use a low temperature (0) so the AI follows instructions strictly.
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.3,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# 3. Initialize FastAPI App
app = FastAPI()

# 4. In-Memory Database for Conversation History
# In a real app, use Redis. Here, a dictionary works for a prototype.
call_memory = {}

SYSTEM_PROMPT = """
You are "DineLine", a friendly phone ordering assistant for a Pizza shop.
Your Menu:
- Pepperoni Pizza ($12)
- Cheese Pizza ($10)
- Coke ($2)

Rules:
1. Keep responses short (under 20 words). This is a voice call.
2. Ask one question at a time.
3. Once the user is done, summarize the order and say "Goodbye".
4. If asked for something not on the menu, politely decline.
"""


@app.post("/voice")
async def voice_handler(request: Request):
    """
    This function is triggered every time the user speaks.
    Twilio sends the data here.
    """

    # Parse Form Data from Twilio
    form_data = await request.form()
    call_sid = form_data.get("CallSid")  # Unique ID for the caller
    user_speech = form_data.get("SpeechResult")  # The text of what the user said

    # Initialize response object (Twilio Markup Language)
    resp = VoiceResponse()

    # Scenario A: New Call (User hasn't spoken yet)
    if call_sid not in call_memory:
        print(f"New Call Started: {call_sid}")
        # Initialize memory with the System Prompt
        call_memory[call_sid] = [SystemMessage(content=SYSTEM_PROMPT)]

        # Greet the user
        greeting = "Thanks for calling AiPhone Pizza. What can I get for you?"
        resp.say(greeting)

        # Save greeting to history so AI knows it said it
        call_memory[call_sid].append(AIMessage(content=greeting))

        # Listen for user input
        resp.gather(input="speech", action="/voice", speechTimeout="auto")
        return Response(content=str(resp), media_type="application/xml")

    # Scenario B: Continuing Conversation
    if user_speech:
        print(f"User said: {user_speech}")

        # 1. Add User's speech to memory
        call_memory[call_sid].append(HumanMessage(content=user_speech))

        # 2. Send full history to Gemini to generate next response
        ai_response = llm.invoke(call_memory[call_sid])
        ai_text = ai_response.content

        print(f"AI replied: {ai_text}")

        # 3. Add AI's reply to memory
        call_memory[call_sid].append(AIMessage(content=ai_text))

        # 4. Speak the response to the user
        resp.say(ai_text)

        # 5. Loop back to listen again (unless conversation is over)
        if "goodbye" in ai_text.lower():
            resp.hangup()
        else:
            resp.gather(input="speech", action="/voice", speechTimeout="auto")

        return Response(content=str(resp), media_type="application/xml")

    # Scenario C: Silence (User didn't say anything)
    resp.say("I didn't hear anything. Are you still there?")
    resp.gather(input="speech", action="/voice")
    return Response(content=str(resp), media_type="application/xml")