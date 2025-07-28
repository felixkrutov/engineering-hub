import os
import json
import google.generativeai as genai
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional

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

class ChatInfo(BaseModel):
    id: str
    title: str

class MessagePart(BaseModel):
    role: str
    parts: List[str]

@app.get("/api/v1/chats", response_model=List[ChatInfo])
async def list_chats():
    chats = []
    os.makedirs(HISTORY_DIR, exist_ok=True)
    for filename in sorted(os.listdir(HISTORY_DIR), reverse=True):
        if filename.endswith(".json"):
            conversation_id = filename[:-5]
            title = "Untitled Chat"
            try:
                filepath = os.path.join(HISTORY_DIR, filename)
                with open(filepath, 'r') as f:
                    history = json.load(f)
                    if history:
                        first_user_message = next((item for item in history if item.get('role') == 'user'), None)
                        if first_user_message and first_user_message.get('parts'):
                           title = first_user_message['parts'][0][:50]
            except (json.JSONDecodeError, IndexError):
                pass
            chats.append(ChatInfo(id=conversation_id, title=title))
    return chats

@app.get("/api/v1/chats/{conversation_id}", response_model=List[dict])
async def get_chat_history(conversation_id: str):
    history_file_path = os.path.join(HISTORY_DIR, f"{conversation_id}.json")
    if not os.path.exists(history_file_path):
        raise HTTPException(status_code=404, detail="Chat history not found.")
    
    try:
        with open(history_file_path, 'r') as f:
            history_data = json.load(f)
        
        # Convert "parts" to "content" for frontend compatibility
        formatted_history = []
        for item in history_data:
            formatted_history.append({
                "role": item.get("role"),
                "content": item.get("parts", [""])[0]
            })
        return formatted_history
    except (json.JSONDecodeError, FileNotFoundError):
        raise HTTPException(status_code=500, detail="Could not read chat history file.")

@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    model = genai.GenerativeModel('gemini-1.5-flash')
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
