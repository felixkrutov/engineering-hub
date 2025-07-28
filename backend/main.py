import os
import json
import google.generativeai as genai
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set!")

genai.configure(api_key=GEMINI_API_KEY)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HISTORY_DIR = "chat_histories"

class ChatRequest(BaseModel):
    message: str
    conversation_id: str

class ChatResponse(BaseModel):
    reply: str

@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    model = genai.GenerativeModel('gemini-2.5-pro')
    conversation_id = request.conversation_id
    history_file_path = os.path.join(HISTORY_DIR, f"{conversation_id}.json")
    
    history = []
    os.makedirs(HISTORY_DIR, exist_ok=True)

    if os.path.exists(history_file_path):
        with open(history_file_path, 'r') as f:
            try:
                history = json.load(f)
            except json.JSONDecodeError:
                history = []

    try:
        chat_session = model.start_chat(history=history)
        
        response = await chat_session.send_message_async(request.message)
        
        history.append({"role": "user", "parts": [request.message]})
        history.append({"role": "model", "parts": [response.text]})

        with open(history_file_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        return ChatResponse(reply=response.text)

    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail="An error occurred with the AI service.")
