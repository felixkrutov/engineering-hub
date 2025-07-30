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
# [ИЗМЕНЕНИЕ 1] Исправляем импорт. Теперь импортируем весь модуль types.
from google.generativeai import types

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

# --- Agent Tools Definition ---
def get_document_content(file_id: str) -> str:
    """
    Reads the content of a document specified by its file_id.
    Returns the text content or an error message.
    """
    logger.info(f"TOOL CALL: get_document_content for file_id: {file_id}")
    try:
        file_info = kb_indexer.get_file_by_id(file_id)
        if not file_info:
            return f"ОШИБКА: Файл с id '{file_id}' не найден в базе знаний."

        file_content = kb_connector.get_file_content(file_id)
        if not file_content:
            return f"ОШИБКА: Не удалось получить содержимое файла '{file_info['name']}'."
        
        parsed_text = parse_document(file_info['name'], file_content, file_info['mime_type'])
        return parsed_text
    except Exception as e:
        logger.error(f"Error in get_document_content tool for file_id {file_id}: {e}", exc_info=True)
        return f"ОШИБКА: Произошла внутренняя ошибка при обработке файла: {e}"


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

# --- Pydantic Models ---

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

class ThinkingStep(BaseModel):
    type: str
    content: str

class Message(BaseModel):
    role: str
    parts: List[str]
    thinking_steps: Optional[List[ThinkingStep]] = None


# --- API Endpoints ---

def load_config() -> AppConfig:
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                data = json.load(f)
                return AppConfig(**data)
    except (json.JSONDecodeError, TypeError):
        pass
    return AppConfig(model_name='gemini-1.5-pro', system_prompt='')

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
            message_data = {
                "role": item.get("role"),
                "content": item.get("parts", [""])[0]
            }
            if 'thinking_steps' in item and item['thinking_steps']:
                message_data['thinking_steps'] = item['thinking_steps']
            formatted_history.append(message_data)

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
    response = ChatResponse(reply="Please use the /api/v1/chat/stream endpoint for chat functionality.", error=True)
    return JSONResponse(status_code=400, content=response.model_dump())

@app.post("/api/v1/chat/stream")
async def stream_chat(request: ChatRequest) -> StreamingResponse:
    async def event_generator():
        steps_history: List[Dict[str, str]] = []
        conversation_id = request.conversation_id
        history_file_path = os.path.join(HISTORY_DIR, f"{conversation_id}.json")
        os.makedirs(HISTORY_DIR, exist_ok=True)
        final_answer_text = "Произошла ошибка при обработке ответа."

        async def yield_and_store(step_data: Dict[str, str]):
            steps_history.append(step_data)
            yield f"data: {json.dumps(step_data)}\n\n"
            await asyncio.sleep(0.1)

        try:
            config = load_config()
            model = genai.GenerativeModel(
                model_name=config.model_name,
                system_instruction=config.system_prompt,
                tools=[get_document_content]
            )
            chat = model.start_chat(enable_automatic_function_calling=False)
            
            history = []
            if os.path.exists(history_file_path):
                with open(history_file_path, 'r') as f:
                    history = json.load(f)

            initial_prompt = request.message
            if request.file_id:
                initial_prompt += f"\n\n[Контекст файла: для анализа файла используй инструмент get_document_content с file_id='{request.file_id}']"
            
            await yield_and_store({'type': 'thought', 'content': 'Отправляю первоначальный запрос модели...'})
            response = chat.send_message(initial_prompt)

            while True:
                # В новых версиях лучше проверять, что candidate существует
                if not response.candidates:
                    final_answer_text = "Модель не вернула кандидатов в ответе."
                    break

                part = response.candidates[0].content.parts[0]
                
                if part.function_call.name:
                    fc = part.function_call
                    await yield_and_store({'type': 'thought', f'content': f"Модель решила вызвать инструмент `{fc.name}` с аргументами: {dict(fc.args)}"})
                    
                    if fc.name == 'get_document_content':
                        tool_result = get_document_content(file_id=fc.args['file_id'])
                    else:
                        tool_result = f"Ошибка: Неизвестный инструмент '{fc.name}'."

                    await yield_and_store({'type': 'tool_result', 'content': tool_result})
                    
                    # [ИЗМЕНЕНИЕ 2] Используем types.Part и types.FunctionResponse для отправки результата
                    response = chat.send_message(
                        content=types.Part(function_response=types.FunctionResponse(
                            name=fc.name,
                            response={"content": tool_result}
                        ))
                    )
                elif part.text:
                    final_answer_text = part.text
                    break
                else:
                    final_answer_text = "Модель вернула пустой ответ."
                    break

        except Exception as e:
            logger.error(f"Error during agent loop for conversation {conversation_id}: {e}", exc_info=True)
            error_content = f"Критическая ошибка в цикле агента: {str(e)}"
            await yield_and_store({'type': 'error', 'content': error_content})
            final_answer_text = "К сожалению, произошла ошибка. Не удалось завершить мыслительный процесс."
        
        user_message = Message(role="user", parts=[request.message])
        model_message = Message(
            role="model",
            parts=[final_answer_text],
            thinking_steps=[ThinkingStep(**step) for step in steps_history]
        )
        
        history.append(user_message.model_dump(exclude_none=True))
        history.append(model_message.model_dump(exclude_none=True))

        with open(history_file_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        if len(history) == 2:
             try:
                title_prompt = f"Summarize the following conversation in 5 words or less. Crucially, you must respond in the same language as the conversation. This will be used as a chat title. Do not use quotation marks.\n\nUser: {request.message}\nAI: {final_answer_text}\n\nTitle:"
                title_model = genai.GenerativeModel('gemini-1.5-flash')
                title_response = title_model.generate_content(title_prompt)
                chat_title = title_response.text.strip().replace('"', '')
                title_file_path = os.path.join(HISTORY_DIR, f"{conversation_id}.title.txt")
                with open(title_file_path, 'w') as f:
                    f.write(chat_title)
             except Exception as e:
                logger.warning(f"An error occurred during title generation: {e}")

        await yield_and_store({'type': 'final_answer', 'content': final_answer_text})

    return StreamingResponse(event_generator(), media_type="text/event-stream")
