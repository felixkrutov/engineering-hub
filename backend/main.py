from fastapi import FastAPI, APIRouter
from pydantic import BaseModel
from typing import Optional

app = FastAPI()
router = APIRouter(prefix="/mossaassistant/api")

class ChatRequest(BaseModel):
    user_message: Optional[str] = None
    chat_id: Optional[int] = None

class ThemeRequest(BaseModel):
    theme: str

@router.get("/chats")
async def get_chats():
    return {"chats": []}

@router.get("/chats/{chat_id}/messages")
async def get_messages(chat_id: int):
    return {"messages": []}

@router.put("/user/theme")
async def update_theme(request: ThemeRequest):
    return {"status": "ok"}

@router.post("/chat")
async def handle_chat(request: ChatRequest):
    response = {
        "ai_response": f"You said: {request.user_message}",
        "is_new_chat": False
    }
    return response

app.include_router(router)
