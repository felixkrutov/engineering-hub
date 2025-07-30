import logging
import os
import json
import asyncio
import google.generativeai as genai
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
from apscheduler.schedulers.background import BackgroundScheduler
from google.generativeai import types

from kb_service.connector import MockConnector
from kb_service.yandex_connector import YandexDiskConnector
from kb_service.indexer import KnowledgeBaseIndexer
from kb_service.parser import parse_document

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

YANDEX_TOKEN = os.getenv("YANDEX_DISK_API_TOKEN")
if YANDEX_TOKEN:
    kb_connector = YandexDiskConnector(token=YANDEX_TOKEN)
else:
    kb_connector = MockConnector()
kb_indexer = KnowledgeBaseIndexer(connector=kb_connector)
scheduler = BackgroundScheduler()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set!")
genai.configure(api_key=GEMINI_API_KEY)

def get_document_content(file_id: str) -> str:
    logger.info(f"TOOL CALL: get_document_content for file_id: {file_id}")
    try:
        file_info = kb_indexer.get_file_by_id(file_id)
        if not file_info: return f"ОШИБКА: Файл с id '{file_id}' не найден."
        file_content = kb_connector.get_file_content(file_id)
        if not file_content: return f"ОШИБКА: Не удалось получить содержимое файла '{file_info['name']}'."
        return parse_document(file_info['name'], file_content, file_info['mime_type'])
    except Exception as e:
        logger.error(f"Error in get_document_content for file_id {file_id}: {e}", exc_info=True)
        return f"ОШИБКА: Внутренняя ошибка при обработке файла: {e}"

app = FastAPI()

def update_kb_index():
    kb_indexer.build_index()

@app.on_event("startup")
def startup_event():
    update_kb_index()
    scheduler.add_job(update_kb_index, "interval", hours=1)
    scheduler.start()

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

HISTORY_DIR = "chat_histories"
CONFIG_FILE = "config.json"

class AppConfig(BaseModel):
    model_name: str
    system_prompt: str
class ChatRequest(BaseModel):
    message: str
    conversation_id: str
    file_id: Optional[str] = None
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

def load_config() -> AppConfig:
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f: return AppConfig(**json.load(f))
    except (json.JSONDecodeError, TypeError): pass
    return AppConfig(model_name='gemini-1.5-pro', system_prompt='')
def save_config(config: AppConfig):
    with open(CONFIG_FILE, 'w') as f: json.dump(config.model_dump(), f, indent=2)

@app.get("/api/v1/config", response_model=AppConfig)
async def get_config(): return load_config()
@app.post("/api/v1/config")
async def set_config(config: AppConfig):
    save_config(config)
    return {"status": "success"}

# ... (остальные эндпоинты, они без изменений) ...
@app.get("/api/kb/search")
async def search_kb(query: str) -> List[Dict[str, str]]:
    return kb_indexer.search(query)

@app.get("/api/kb/file/{file_id:path}")
async def get_kb_file(file_id: str) -> StreamingResponse:
    content = kb_connector.get_file_content(file_id)
    if content is None: raise HTTPException(404, "File not found")
    file_meta = kb_indexer.get_file_by_id(file_id)
    media_type = file_meta.get('mime_type', 'application/octet-stream') if file_meta else 'application/octet-stream'
    return StreamingResponse(iter([content]), media_type=media_type)

@app.get("/api/v1/chats", response_model=List[ChatInfo])
async def list_chats():
    chats = []
    os.makedirs(HISTORY_DIR, exist_ok=True)
    for filename in os.listdir(HISTORY_DIR):
        if not filename.endswith(".json"): continue
        conversation_id = filename[:-5]
        title_path = os.path.join(HISTORY_DIR, f"{conversation_id}.title.txt")
        title = "Новый чат"
        if os.path.exists(title_path):
            with open(title_path, 'r') as f: title = f.read().strip() or title
        else:
            try:
                with open(os.path.join(HISTORY_DIR, filename), 'r') as f:
                    history = json.load(f)
                    if history:
                        first_user_message = next((item for item in history if item.get('role') == 'user'), None)
                        if first_user_message and first_user_message.get('parts'):
                           title = first_user_message['parts'][0][:50]
            except (json.JSONDecodeError, IndexError): pass
        chats.append(ChatInfo(id=conversation_id, title=title))
    return sorted(chats, key=lambda x: os.path.getmtime(os.path.join(HISTORY_DIR, f"{x.id}.json")), reverse=True)

@app.get("/api/v1/chats/{conversation_id}")
async def get_chat_history(conversation_id: str):
    history_file_path = os.path.join(HISTORY_DIR, f"{conversation_id}.json")
    if not os.path.exists(history_file_path): raise HTTPException(404, "Chat history not found.")
    with open(history_file_path, 'r') as f: history_data = json.load(f)
    return history_data

@app.delete("/api/v1/chats/{conversation_id}", status_code=204)
async def delete_chat(conversation_id: str):
    history_file = os.path.join(HISTORY_DIR, f"{conversation_id}.json")
    title_file = os.path.join(HISTORY_DIR, f"{conversation_id}.title.txt")
    if os.path.exists(history_file): os.remove(history_file)
    if os.path.exists(title_file): os.remove(title_file)
    return

@app.put("/api/v1/chats/{conversation_id}")
async def rename_chat(conversation_id: str, request: RenameRequest):
    title_path = os.path.join(HISTORY_DIR, f"{conversation_id}.title.txt")
    if not os.path.exists(os.path.join(HISTORY_DIR, f"{conversation_id}.json")): raise HTTPException(404, "Chat not found.")
    with open(title_path, 'w') as f: f.write(request.new_title)
    return {"status": "success"}
# ... (конец неизменных эндпоинтов) ...


@app.post("/api/v1/chat/stream")
async def stream_chat(request: ChatRequest) -> StreamingResponse:
    async def event_generator():
        steps_history: List[Dict[str, Any]] = []
        conversation_id = request.conversation_id
        history_file_path = os.path.join(HISTORY_DIR, f"{conversation_id}.json")
        os.makedirs(HISTORY_DIR, exist_ok=True)
        final_answer_text = ""

        async def yield_and_store(step_data: Dict[str, Any]):
            steps_history.append(step_data)
            yield f"data: {json.dumps(step_data)}\n\n"
            await asyncio.sleep(0.01)

        try:
            config = load_config()
            model = genai.GenerativeModel(
                model_name=config.model_name,
                system_instruction=config.system_prompt,
                tools=[get_document_content]
            )
            history = []
            if os.path.exists(history_file_path):
                with open(history_file_path, 'r') as f: history = json.load(f)
            
            chat = model.start_chat(history=[msg for msg in history if 'thinking_steps' not in msg])
            
            initial_prompt = request.message
            if request.file_id:
                initial_prompt += f"\n\n[Контекст файла: для анализа файла используй инструмент get_document_content с file_id='{request.file_id}']"
            
            await yield_and_store({'type': 'thought', 'content': 'Отправляю запрос модели...'})
            response_stream = chat.send_message(initial_prompt, stream=True)

            # ИСПОЛЬЗУЕМ ASYNC FOR
            async for chunk in response_stream:
                if not chunk.candidates: continue

                part = chunk.candidates[0].content.parts[0]

                if part.function_call:
                    fc = part.function_call
                    await yield_and_store({'type': 'tool_code', 'content': f"{fc.name}({json.dumps(dict(fc.args), ensure_ascii=False)})"})
                    
                    tool_result = get_document_content(file_id=fc.args['file_id'])
                    
                    await yield_and_store({'type': 'tool_result', 'content': tool_result})

                    response_stream_after_tool = chat.send_message(
                        content=types.Part(function_response=types.FunctionResponse(name=fc.name, response={"content": tool_result})),
                        stream=True
                    )
                    # ИСПОЛЬЗУЕМ ASYNC FOR И ЗДЕСЬ
                    async for tool_chunk in response_stream_after_tool:
                        if tool_chunk.text:
                            final_answer_text += tool_chunk.text
                            await yield_and_store({'type': 'text_chunk', 'content': tool_chunk.text})
                    
                    # После обработки инструмента, цикл должен завершиться, так как мы получили полный ответ
                    break 
                
                elif part.text:
                    final_answer_text += part.text
                    await yield_and_store({'type': 'text_chunk', 'content': part.text})
            
            if not final_answer_text.strip():
                final_answer_text = "Модель не предоставила ответа."

        except Exception as e:
            logger.error(f"Error during agent loop for {conversation_id}: {e}", exc_info=True)
            error_content = f"Критическая ошибка: {str(e)}"
            await yield_and_store({'type': 'error', 'content': error_content})
            final_answer_text = "К сожалению, произошла ошибка."
        
        # Сохранение истории
        user_message = Message(role="user", parts=[request.message])
        model_message = Message(role="model", parts=[final_answer_text], thinking_steps=[ThinkingStep(**step) for step in steps_history])
        history.extend([user_message.model_dump(exclude_none=True), model_message.model_dump(exclude_none=True)])
        with open(history_file_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        if len(history) == 2:
             try:
                title_model = genai.GenerativeModel('gemini-1.5-flash')
                title_response = await title_model.generate_content_async(f"Summarize in 5 words or less in the same language: User: {request.message}\nAI: {final_answer_text}")
                chat_title = title_response.text.strip().replace('"', '')
                with open(os.path.join(HISTORY_DIR, f"{conversation_id}.title.txt"), 'w', encoding='utf-8') as f:
                    f.write(chat_title)
             except Exception as e:
                logger.warning(f"Title generation failed: {e}")

        await yield_and_store({'type': 'final_answer', 'content': final_answer_text})

    return StreamingResponse(event_generator(), media_type="text/event-stream")
