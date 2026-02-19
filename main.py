import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Form
from fastapi.responses import Response
from twilio.twiml.voice_response import VoiceResponse
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

# --- LANGGRAPH IMPORTS ---
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver

# 1. Load Environment Variables
load_dotenv()

# 2. Setup Google Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.1,  # Low temperature so it doesn't hallucinate menu items
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# 3. Define the Menu & Rules (This guides the AI)
SYSTEM_PROMPT = """You are DineLine, an AI phone ordering assistant.
Your Menu:
- Pepperoni Pizza ($12) - must specify size (Small/Large)
- Cheese Pizza ($10) - must specify size (Small/Large)
- Coke ($2)

Rules:
1. Keep responses under 20 words. This is a voice call.
2. If a user orders a pizza, ALWAYS ask what size they want if they didn't specify.
3. If they ask for something not on the menu, politely decline.
4. Once they say they are done ordering, summarize the order, give the total price, and say "Goodbye".
5. Do not use markdown (* or **). Speak in plain text.
"""


# --- 4. BUILD THE LANGGRAPH ---

# Node Function: This is the worker that calls the LLM
def call_model(state: MessagesState):
    # The LLM looks at the entire history in the state and generates a reply
    response = llm.invoke(state["messages"])
    # LangGraph automatically appends this response to the history
    return {"messages": response}


# Create the Graph (The Flowchart)
workflow = StateGraph(MessagesState)
workflow.add_node("agent", call_model)
workflow.add_edge(START, "agent")
workflow.add_edge("agent", END)

# Initialize MemorySaver to persist history between Twilio requests
memory = MemorySaver()
app_graph = workflow.compile(checkpointer=memory)

# --- 5. FASTAPI SERVER ---
app = FastAPI()


@app.post("/voice")
async def voice_handler(request: Request):
    form_data = await request.form()
    call_sid = form_data.get("CallSid")  # Unique ID for this specific phone call
    user_speech = form_data.get("SpeechResult")

    # Tell LangGraph to use the CallSid to find the correct memory thread
    config = {"configurable": {"thread_id": call_sid}}

    resp = VoiceResponse()

    # SCENARIO A: The call just connected (No user speech yet)
    if not user_speech:
        print(f"\n--- New Call Started: {call_sid} ---")

        # Inject the System Prompt silently into this call's memory
        app_graph.invoke(
            {"messages": [SystemMessage(content=SYSTEM_PROMPT)]},
            config=config
        )

        greeting = "Welcome to DineLine Pizza! What would you like to order today?"

        # Add the greeting to memory so the AI knows it already said hello
        app_graph.invoke(
            {"messages": [{"role": "assistant", "content": greeting}]},
            config=config
        )

        resp.say(greeting)
        resp.gather(input="speech", action="/voice", speechTimeout="auto")
        return Response(content=str(resp), media_type="application/xml")

    # SCENARIO B: The user spoke
    print(f"User said: {user_speech}")

    # Send the user's speech into the LangGraph. It remembers everything before this.
    events = app_graph.invoke(
        {"messages": [HumanMessage(content=user_speech)]},
        config=config
    )

    # Extract the AI's latest reply from the state
    ai_response = events["messages"][-1].content
    print(f"AI replied: {ai_response}")

    # Speak the reply over the phone
    resp.say(ai_response)

    # Check if the AI ended the conversation
    if "goodbye" in ai_response.lower():
        resp.hangup()
    else:
        # If not done, listen again
        resp.gather(input="speech", action="/voice", speechTimeout="auto")

    return Response(content=str(resp), media_type="application/xml")