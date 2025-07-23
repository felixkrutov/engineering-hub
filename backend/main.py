from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class ChatRequest(BaseModel):
    user_message: str | None = None
    chat_id: int | None = None

@app.post("/mossaassistant/api/chat")
async def handle_chat(request: ChatRequest):
    response = {
        "ai_response": f"You said: {request.user_message}",
        "is_new_chat": False
    }
    return response
