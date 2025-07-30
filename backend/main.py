import logging
import os
import json
import uuid
import asyncio
from datetime import datetime
import google.generativeai as genai
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
from apscheduler.schedulers.background import BackgroundScheduler
from google.api_core.exceptions import ResourceExhausted

from kb_service.connector import MockConnector
from kb_service.yandex_connector import YandexDiskConnector
from kb_service.indexer import KnowledgeBaseIndexer
from kb_service.parser import parse_document

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# --- Knowledge Base Services Initialization ---
YANDEX_TOKEN = os.getenv("YANDEX_DISK_API_TOKEN")

if YANDEX_TOKEN:
    logging.info("YANDEX_DISK_API_TOKEN found. Initializing YandexDiskConnector.")
    kb_connector = YandexDiskConnector(token=YANDEX_TOKEN)
else:
    logging.info("YANDEX_DISK_API_TOKEN not found. Initializing MockConnector as a fallback.")
    kb_connector = MockConnector()

kb_indexer = KnowledgeBaseIndexer(connector=kb_connector)

# Instantiate Scheduler
scheduler = BackgroundScheduler()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set!")

genai.configure(api_key=GEMINI_API_KEY)

app = FastAPI()

def update_kb_index() -> None:
    kb_indexer.build_index()

@app.on_event("startup")
def startup_event():
    logging.info("Application startup: Initializing services...")
    update_kb_index()
    scheduler.add_job(update_kb_index, "interval", hours=1)
    scheduler.start()
    logging.info("Application startup: Services initialized and scheduler started.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HISTORY_DIR = "chat_histories"
CONFIG_FILE = "config.json"

class AppConfig(BaseModel):
    model_name: str
    system_prompt: str

class ChatRequest(BaseModel):
    message: str
    conversation_id: str
    file_id: Optional[str] = None

class ChatResponse(BaseModel):
    reply: str
    error: bool = False

class ChatInfo(BaseModel):
    id: str
    title: str
    
class RenameRequest(BaseModel):
    new_title: str

def load_config() -> AppConfig:
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                data = json.load(f)
                return AppConfig(**data)
    except (json.JSONDecodeError, TypeError):
        pass
    return AppConfig(model_name='gemini-1.5-flash', system_prompt='')

def save_config(config: AppConfig):
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config.model_dump(), f, indent=2)

@app.get("/api/v1/config", response_model=AppConfig)
async def get_config():
    return load_config()

@app.post("/api/v1/config", status_code=status.HTTP_200_OK)
async def set_config(config: AppConfig):
    try:
        save_config(config)
        return {"status": "success", "message": "Configuration saved."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/kb/search")
async def search_kb(query: str) -> List[Dict[str, str]]:
    logging.info(f"Received search request with query: '{query}'")
    results = kb_indexer.search(query)
    return results

@app.get("/api/kb/file/{file_id:path}")
async def get_kb_file(file_id: str) -> StreamingResponse:
    logging.info(f"Received request for file content: {file_id}")
    content = kb_connector.get_file_content(file_id)

    if content is None:
        logging.error(f"File not found: {file_id}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found")

    file_meta = kb_indexer.get_file_by_id(file_id)
    media_type = "application/octet-stream"
    if file_meta:
        media_type = file_meta.get('mime_type', 'application/octet-stream')
    
    return StreamingResponse(iter([content]), media_type=media_type)

@app.get("/api/v1/chats", response_model=List[ChatInfo])
async def list_chats():
    chats = []
    os.makedirs(HISTORY_DIR, exist_ok=True)
    for filename in os.listdir(HISTORY_DIR):
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


@app.post("/api/v1/chat")
async def chat(request: ChatRequest) -> Any:
    config = load_config()
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

    final_message = request.message
    file_id = request.file_id
    
    if file_id:
        try:
            logger.info(f"RAG request received for file_id: {file_id}")
            file_info = kb_indexer.get_file_by_id(file_id)
            if not file_info:
                raise HTTPException(status_code=404, detail=f"File with id {file_id} not found in index.")
            
            file_content = kb_connector.get_file_content(file_id)
            if not file_content:
                raise HTTPException(status_code=404, detail=f"Content for file id {file_id} could not be retrieved.")
            
            document_text = parse_document(file_info['name'], file_content, file_info['mime_type'])
            
            final_message = f"Context from file '{file_info['name']}':\n\n{document_text}\n\nUser query: {request.message}"
            logger.info("Successfully pre-pended document context to user message.")
        
        except Exception as e:
            logger.error(f"Error processing RAG request for file_id {file_id}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to process document for RAG.")

    try:
        model = genai.GenerativeModel(
            model_name=config.model_name,
            system_instruction=config.system_prompt if config.system_prompt.strip() else None
        )
        chat_session = model.start_chat(history=history)
        response = await chat_session.send_message_async(final_message)
        
        history.append({"role": "user", "parts": [request.message]})
        history.append({"role": "model", "parts": [response.text]})

        with open(history_file_path, 'w') as f:
            json.dump(history, f, indent=2)

        if len(history) == 2:
            try:
                title_prompt = f"Summarize the following conversation in 5 words or less. Crucially, you must respond in the same language as the conversation. This will be used as a chat title. Do not use quotation marks.\n\nUser: {request.message}\nAI: {response.text}\n\nTitle:"
                title_model = genai.GenerativeModel('gemini-1.5-flash')
                title_response = title_model.generate_content(title_prompt)
                chat_title = title_response.text.strip().replace('"', '')
                title_file_path = os.path.join(HISTORY_DIR, f"{conversation_id}.title.txt")
                with open(title_file_path, 'w') as f:
                    f.write(chat_title)
            except Exception as e:
                logger.warning(f"An error occurred during title generation: {e}")
        
        return ChatResponse(reply=response.text)

    except ResourceExhausted as e:
        logger.error(f"Google API quota exceeded: {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "reply": "Не удалось обработать запрос: исчерпан лимит запросов к API. Пожалуйста, попробуйте позже или проверьте ваш план и биллинг.",
                "error": True
            }
        )
    except Exception as e:
        logger.error(f"An unexpected error occurred in chat endpoint: {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "reply": "Произошла непредвиденная ошибка на сервере. Пожалуйста, попробуйте еще раз.",
                "error": True
            }
        )

@app.post("/api/v1/chat/stream")
async def stream_chat(request: ChatRequest) -> StreamingResponse:
    async def event_generator():
        # Step 1: Acknowledge and think
        yield f"data: {json.dumps({'type': 'thought', 'content': 'Задача получена. Начинаю анализ.'})}\n\n"
        await asyncio.sleep(1) # Simulate work

        # Step 2: Simulate a tool call
        yield f"data: {json.dumps({'type': 'tool_call', 'content': 'Использую инструмент: get_file_content'})}\n\n"
        await asyncio.sleep(1.5)

        # Step 3: Simulate a tool result (error)
        yield f"data: {json.dumps({'type': 'tool_result', 'content': 'Ошибка: Файл слишком большой. Лимит токенов превышен.'})}\n\n"
        await asyncio.sleep(1)

        # Step 4: Final thought before answering
        yield f"data: {json.dumps({'type': 'thought', 'content': 'Не удалось обработать файл. Формулирую ответ для пользователя.'})}\n\n"
        await asyncio.sleep(0.5)

        # Step 5: The final answer
        final_answer_text = "К сожалению, я не могу напрямую проанализировать этот файл, так как он слишком большой."
        yield f"data: {json.dumps({'type': 'final_answer', 'content': final_answer_text})}\n\n"
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")
