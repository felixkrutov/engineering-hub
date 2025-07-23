from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

app = FastAPI()

class ChatRequest(BaseModel):
    user_message: Optional[str] = None
    chat_id: Optional[int] = None

class ThemeRequest(BaseModel):
    theme: str

@app.get("/mossaassistant/api/chats")
async def get_chats():
    return {"chats": []}

@app.get("/mossaassistant/api/chats/{chat_id}/messages")
async def get_messages(chat_id: int):
    return {"messages": []}

@app.put("/mossaassistant/api/user/theme")
async def update_theme(request: ThemeRequest):
    return {"status": "ok"}

@app.post("/mossaassistant/api/chat")
async def handle_chat(request: ChatRequest):
    response = {
        "ai_response": f"You said: {request.user_message}",
        "is_new_chat": False
    }
    return response
