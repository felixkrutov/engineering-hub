import logging
import os
import json
import uuid
import re
import asyncio
from datetime import datetime
import google.generativeai as genai
import google.generativeai.protos as gap
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
scheduler = BackgroundScheduler()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set!")

genai.configure(api_key=GEMINI_API_KEY)

# --- Agent Tools Definition ---
def analyze_document(file_id: str, query: str) -> str:
    logger.info(f"TOOL CALL: analyze_document for file_id: {file_id} with query: '{query}'")
    try:
        results = kb_indexer.search(query=query, file_id=file_id)
        if not results:
            file_info = kb_indexer.get_file_by_id(file_id)
            file_name = file_info['name'] if file_info else file_id
            return f"Внутри файла '{file_name}' по вашему запросу '{query}' ничего не найдено."
        formatted_results = [f"--- Результат поиска №{i+1} (из файла: {chunk['file_name']}) ---\n{chunk['text']}\n" for i, chunk in enumerate(results)]
        return "\n".join(formatted_results)
    except Exception as e:
        logger.error(f"Error in analyze_document tool for file_id {file_id}: {e}", exc_info=True)
        return f"ОШИБКА: Произошла внутренняя ошибка при поиске по файлу: {e}"

def search_knowledge_base(query: str) -> str:
    logger.info(f"TOOL CALL: search_knowledge_base with query: '{query}'")
    results = kb_indexer.search(query)
    if not results:
        return "По вашему запросу в базе знаний ничего не найдено."
    formatted_results = [f"--- Результат поиска №{i+1} (из файла: {chunk['file_name']}) ---\n{chunk['text']}\n" for i, chunk in enumerate(results)]
    return "\n".join(formatted_results)


app = FastAPI()

def update_kb_index() -> None:
    kb_indexer.build_index()

@app.on_event("startup")
def startup_event():
    logging.info("Application startup: Initializing services...")
    update_kb_index()
    scheduler.add_job(update_kb_index, "interval", hours=1, id="update_kb_index_job", replace_existing=True)
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
CONTROLLER_SYSTEM_PROMPT = "You are a helpful assistant."

# --- Pydantic Models ---
class AgentSettings(BaseModel):
    model_name: str
    system_prompt: str

class AppConfig(BaseModel):
    executor: AgentSettings
    controller: AgentSettings

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


# --- API Endpoints ---
# --- INSTRUCTION: More robust config loading ---
def load_config() -> AppConfig:
    default_config = AppConfig(
        executor=AgentSettings(model_name='gemini-1.5-pro', system_prompt=''),
        controller=AgentSettings(model_name='gpt-4o-mini', system_prompt=CONTROLLER_SYSTEM_PROMPT)
    )
    if not os.path.exists(CONFIG_FILE):
        return default_config
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return AppConfig.model_validate(data)
    except Exception as e:
        logger.warning(f"Could not load or validate config file due to: {e}. Deleting corrupt file and using defaults.")
        try:
            os.remove(CONFIG_FILE) # Delete the corrupt file
        except OSError as del_e:
            logger.error(f"Failed to delete corrupt config file: {del_e}")
        return default_config

def save_config(config: AppConfig):
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config.model_dump(), f, indent=2, ensure_ascii=False)

@app.get("/api/v1/config", response_model=AppConfig)
async def get_config():
    return load_config()

@app.post("/api/v1/config", status_code=status.HTTP_200_OK)
async def set_config(config: AppConfig):
    save_config(config)
    return {"status": "success", "message": "Configuration saved."}

@app.get("/api/kb/files", response_model=List[Dict])
async def get_all_kb_files():
    return kb_indexer.get_all_files()

@app.post("/api/v1/chat/stream")
async def stream_chat(request: ChatRequest) -> StreamingResponse:
    async def event_generator():
        steps_history: List[Dict[str, Any]] = []
        conversation_id = request.conversation_id
        history_file_path = os.path.join(HISTORY_DIR, f"{conversation_id}.json")
        os.makedirs(HISTORY_DIR, exist_ok=True)
        final_answer_text = "Произошла ошибка при обработке ответа."

        history = []
        if os.path.exists(history_file_path):
            with open(history_file_path, 'r', encoding='utf-8') as f:
                history = json.load(f)

        try:
            config = load_config()
            model_kwargs = {
                'model_name': config.executor.model_name,
                'tools': [analyze_document, search_knowledge_base]
            }
            if config.executor.system_prompt and config.executor.system_prompt.strip():
                model_kwargs['system_instruction'] = config.executor.system_prompt
            
            model = genai.GenerativeModel(**model_kwargs)
            cleaned_history = [{"role": msg["role"], "parts": msg["parts"]} for msg in history]
            chat_session = model.start_chat(history=cleaned_history)
            initial_prompt = request.message
            if request.file_id:
                initial_prompt += f"\n\n[ИНСТРУКЦИЯ ДЛЯ АГЕНТА]\nПользователь сфокусирован на конкретном файле. Его ID: '{request.file_id}'. Для ответа на вопрос пользователя, ты ДОЛЖЕН использовать инструмент `analyze_document`. Передай этот ID в аргумент `file_id` и извлеки поисковый запрос пользователя из его сообщения для аргумента `query`."
            
            step_data = {'type': 'thought', 'content': 'Обдумываю ваш запрос...'}
            steps_history.append(step_data)
            yield f"data: {json.dumps(step_data)}\n\n"
            
            response = await chat_session.send_message_async(initial_prompt)

            while True:
                if not response.candidates:
                    final_answer_text = "Модель не вернула кандидатов в ответе. Возможно, сработал защитный фильтр."
                    break

                part = response.candidates[0].content.parts[0]
                
                if hasattr(part, 'function_call') and part.function_call.name:
                    fc = part.function_call
                    human_readable_action = ""
                    if fc.name == 'search_knowledge_base':
                        human_readable_action = f"Ищу информацию по запросу «{fc.args.get('query')}» во всей базе знаний..."
                    elif fc.name == 'analyze_document':
                        file_info = kb_indexer.get_file_by_id(fc.args.get('file_id'))
                        file_name = file_info['name'] if file_info else fc.args.get('file_id')
                        human_readable_action = f"Анализирую содержимое файла «{file_name}» по запросу «{fc.args.get('query')}»..."

                    if human_readable_action:
                        step_data = {'type': 'thought', 'content': human_readable_action}
                        steps_history.append(step_data)
                        yield f"data: {json.dumps(step_data)}\n\n"
                    
                    tool_result = ""
                    if fc.name == 'analyze_document':
                        tool_result = analyze_document(file_id=fc.args.get('file_id'), query=fc.args.get('query'))
                    elif fc.name == 'search_knowledge_base':
                        tool_result = search_knowledge_base(query=fc.args.get('query'))
                    else:
                        tool_result = f"Ошибка: Неизвестный инструмент '{fc.name}'."

                    if "ОШИБКА:" in tool_result or "ничего не найдено" in tool_result:
                        summary_content = "В базе знаний не найдено релевантной информации."
                    else:
                        source_files = list(set(re.findall(r"\(из файла: (.*?)\)", tool_result)))
                        summary_content = f"Найдена релевантная информация в следующих документах: {', '.join(source_files)}." if source_files else "Найдена релевантная информация в базе знаний."

                    step_data = {'type': 'tool_result', 'content': summary_content}
                    steps_history.append(step_data)
                    yield f"data: {json.dumps(step_data)}\n\n"
                    
                    response = await chat_session.send_message_async(
                        gap.Part(function_response=gap.FunctionResponse(name=fc.name, response={'content': tool_result}))
                    )
                elif hasattr(part, 'text') and part.text:
                    final_answer_text = part.text
                    break
                else:
                    final_answer_text = "Модель вернула пустой ответ."
                    break
        
        except Exception as e:
            logger.error(f"Error during agent loop for conversation {conversation_id}: {e}", exc_info=True)
            error_content = f"Критическая ошибка в цикле агента: {str(e)}"
            yield f"data: {json.dumps({'type': 'error', 'content': error_content})}\n\n"
            final_answer_text = "К сожалению, произошла ошибка. Не удалось завершить мыслительный процесс."
        
        user_message = Message(role="user", parts=[request.message])
        model_message = Message(role="model", parts=[final_answer_text], thinking_steps=[ThinkingStep(**step) for step in steps_history])
        
        history.append(user_message.model_dump(exclude_none=True))
        history.append(model_message.model_dump(exclude_none=True))

        # --- INSTRUCTION: Robust chat history saving ---
        try:
            with open(history_file_path, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"CRITICAL: Failed to save chat history for {conversation_id}: {e}", exc_info=True)
        
        if len(history) == 2:
             try:
                title_prompt = f"Summarize the following conversation in 5 words or less in Russian. User: {request.message}\nAI: {final_answer_text}\n\nTitle:"
                title_model = genai.GenerativeModel('gemini-1.5-flash')
                title_response = await title_model.generate_content_async(title_prompt)
                chat_title = title_response.text.strip().replace('"', '')
                title_file_path = os.path.join(HISTORY_DIR, f"{conversation_id}.title.txt")
                with open(title_file_path, 'w', encoding='utf-8') as f:
                    f.write(chat_title)
             except Exception as e:
                logger.warning(f"An error occurred during title generation: {e}")

        final_data = {'type': 'final_answer', 'content': final_answer_text}
        yield f"data: {json.dumps(final_data)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

# Other endpoints like get_chat_history, delete_chat, rename_chat, etc. would follow here
# (omitted for brevity but should be in the final file)
@app.get("/api/v1/chats/{conversation_id}")
async def get_chat_history(conversation_id: str):
    history_file_path = os.path.join(HISTORY_DIR, f"{conversation_id}.json")
    if not os.path.exists(history_file_path):
        raise HTTPException(status_code=404, detail="Chat history not found.")
    try:
        with open(history_file_path, 'r', encoding='utf-8') as f:
            history_data = json.load(f)
        formatted_history = []
        for item in history_data:
            message_data = {"role": item.get("role"), "content": item.get("parts", [""])[0] }
            if 'thinking_steps' in item and item['thinking_steps']:
                message_data['thinking_steps'] = item['thinking_steps']
            formatted_history.append(message_data)
        return formatted_history
    except (json.JSONDecodeError, FileNotFoundError):
        raise HTTPException(status_code=500, detail="Could not read or parse chat history file.")

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

@app.put("/api/v1/chats/{conversation_id}")
async def rename_chat(conversation_id: str, request: RenameRequest):
    history_file_path = os.path.join(HISTORY_DIR, f"{conversation_id}.json")
    if not os.path.exists(history_file_path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat not found, cannot rename.")
    title_file_path = os.path.join(HISTORY_DIR, f"{conversation_id}.title.txt")
    try:
        with open(title_file_path, 'w', encoding='utf-8') as f:
            f.write(request.new_title)
    except OSError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error writing title file: {e}")
    return {"status": "success", "message": "Chat renamed"}
