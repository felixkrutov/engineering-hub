import os
import json
import google.generativeai as genai
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status
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
    
class RenameRequest(BaseModel):
    new_title: str

@app.get("/api/v1/chats", response_model=List[ChatInfo])
async def list_chats():
    chats = []
    os.makedirs(HISTORY_DIR, exist_ok=True)
    for filename in sorted(os.listdir(HISTORY_DIR)):
        if filename.endswith(".json"):
            conversation_id = filename[:-5]
            title_path = os.path.join(HISTORY_DIR, f"{conversation_id}.title.txt")
            title = "Untitled Chat"

            if os.path.exists(title_path):
                with open(title_path, 'r') as f:
                    title = f.read().strip() or title
            else:
                try:
                    with open(os.path.join(HISTORY_DIR, filename), 'r') as f:
                        history = json.load(f)
                        if history:
                            first_user_message = next((item for item in history if item.get('role') == 'user'), None)
                            if first_user_message and first_user_message.get('parts'):
                               title = first_user_message['parts'][0][:50]
                except (json.JSONDecodeError, IndexError):
                    pass
            chats.append(ChatInfo(id=conversation_id, title=title))
    return sorted(chats, key=lambda x: x.id, reverse=True)


@app.get("/api/v1/chats/{conversation_id}")
async def get_chat_history(conversation_id: str):
    history_file_path = os.path.join(HISTORY_DIR, f"{conversation_id}.json")
    if not os.path.exists(history_file_path):
        raise HTTPException(status_code=404, detail="Chat history not found.")
    
    try:
        with open(history_file_path, 'r') as f:
            history_data = json.load(f)
        
        formatted_history = []
        for item in history_data:
            formatted_history.append({
                "role": item.get("role"),
                "content": item.get("parts", [""])[0]
            })
        return formatted_history
    except (json.JSONDecodeError, FileNotFoundError):
        raise HTTPException(status_code=500, detail="Could not read chat history file.")

@app.delete("/api/v1/chats/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_chat(conversation_id: str):
    history_file_path = os.path.join(HISTORY_DIR, f"{conversation_id}.json")
    title_file_path = os.path.join(HISTORY_DIR, f"{conversation_id}.title.txt")

    if not os.path.exists(history_file_path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat not found")
    
    try:
        os.remove(history_file_path)
        if os.path.exists(title_file_path):
            os.remove(title_file_path)
    except OSError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error deleting chat files: {e}")
    return

@app.put("/api/v1/chats/{conversation_id}")
async def rename_chat(conversation_id: str, request: RenameRequest):
    history_file_path = os.path.join(HISTORY_DIR, f"{conversation_id}.json")
    title_file_path = os.path.join(HISTORY_DIR, f"{conversation_id}.title.txt")

    if not os.path.exists(history_file_path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat not found, cannot rename.")
    
    try:
        with open(title_file_path, 'w') as f:
            f.write(request.new_title)
    except OSError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error writing title file: {e}")
    
    return {"status": "success", "message": "Chat renamed"}


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
