# Real-Time AI Voice Ordering Agent

An enterprise-grade, asynchronous AI voice assistant built for the hospitality industry. DineLine orchestrates real-time bidirectional audio streaming to handle customer phone calls, search JSON-based restaurant menus, and manage shopping carts with deterministic accuracy.

## 🚀 System Architecture & Flow
Unlike basic chat wrappers, this system operates on a highly concurrent event loop using **FastAPI** and **WebSockets**, bridging raw audio between telephony networks and AI models with sub-500ms latency.

1. **Telephony Layer:** Twilio captures the phone call and opens a WebSocket stream (`wss://`).
2. **STT (Speech-to-Text):** Raw audio bytes are streamed in real-time to Deepgram for lightning-fast transcription and Voice Activity Detection (VAD).
3. **State Machine (The Brain):** LangGraph manages the conversational memory and routes user intent to Google Gemini. 
4. **Tool Execution:** To prevent LLM hallucinations, Gemini is isolated from business logic. It utilizes strict Python tool-calling to query the `menu.json` database, calculate prices, and manipulate the cart.
5. **TTS (Text-to-Speech):** AI-generated text is streamed to ElevenLabs, returning mu-law encoded audio back down the Twilio WebSocket to the customer.

## 🛠️ Tech Stack
* **Backend Framework:** Python, FastAPI, Uvicorn
* **Agentic Framework:** LangGraph, LangChain
* **LLM:** Google Gemini 2.5 Flash
* **Telephony & Audio:** Twilio (Media Streams), Deepgram (Real-time STT), ElevenLabs (Ultra-low latency TTS)
* **Architecture:** Event-Driven, WebSocket Streams, Retrieval-Augmented Generation (RAG)

## ✨ Key Features
* **Deterministic Guardrails:** The LLM never calculates prices or guesses inventory. It strictly relies on a deterministic Python backend to execute `search_menu`, `add_to_cart`, and `remove_from_cart` tools.
* **Separation of Concerns (SoC):** The audio-streaming infrastructure (`main.py`) is completely decoupled from the AI and business logic (`agent.py`).
* **Active Memory Leak Prevention:** Graceful handling of WebSocket disconnections and background thread execution for Twilio TwiML updates.
* **Emergency Human Handoff:** Instantly bridges the live phone call to a human manager if the AI detects user frustration or receives a direct request.

## 📁 Project Structure
```text
aiPhone_proto/
├── main.py          # FastAPI server, WebSocket routing, STT/TTS API integrations
├── agent.py         # LangGraph state machine, Gemini LLM, Python tool logic
├── test_bot.py      # CLI simulator for rapid, cost-free LLM logic testing
├── menu.json        # Hierarchical NoSQL-style database for the restaurant menu
├── .env             # Environment variables and API keys
└── requirements.txt # Python dependencies
```

## 💻 Local Setup & Installation

### 1. Prerequisites
You will need API keys for the following services:
* Google Gemini
* Twilio
* Deepgram
* ElevenLabs

### 2. Install Dependencies
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
pip install fastapi uvicorn websockets twilio langchain-google-genai langgraph httpx python-dotenv
```

### 3. Environment Variables
Create a `.env` file in the root directory and add your keys. **Do not commit this file to GitHub.**
```env
GOOGLE_API_KEY=your_gemini_key
TWILIO_ACCOUNT_SID=your_twilio_sid
TWILIO_AUTH_TOKEN=your_twilio_token
DEEPGRAM_API_KEY=your_deepgram_key
ELEVENLABS_API_KEY=your_elevenlabs_key
ELEVENLABS_VOICE_ID=your_chosen_voice_id
STAFF_PHONE_NUMBER=+1234567890 
```

### 4. Running the Project

**Option A: Run the CLI Simulator (No audio, tests logic only)**
Great for testing graph routing and prompt engineering without consuming audio API credits.
```bash
python test_bot.py
```

**Option B: Run the Live Voice Server**

1. Start the FastAPI server:
```bash
uvicorn main:app --reload
```

2. Expose your local port via ngrok:
```bash
ngrok http 8000
```

3. Copy the secure `https://...` ngrok URL and paste it into your Twilio Phone Number's active Webhook configuration, appending `/voice` to the end (e.g., `https://your-url.ngrok-free.app/voice`).

4. Call your Twilio number!

## 🔮 Future Roadmap
- [ ] **State Persistence:** Migrate LangGraph MemorySaver to a PostgreSQL/Redis Checkpointer for fault tolerance.
- [ ] **Semantic Caching:** Implement Redis caching for frequent queries (e.g., "What are your hours?") to reduce LLM API latency and costs.
- [ ] **POS Integration:** Build a webhook to package the final LangGraph cart state into a structured payload for injection into Square or Lightspeed REST APIs.
