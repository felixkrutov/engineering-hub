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
from google.api_core.exceptions import ResourceExhausted, InternalServerError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import AsyncOpenAI
import redis

from kb_service.connector import MockConnector
from kb_service.yandex_connector import YandexDiskConnector
from kb_service.indexer import KnowledgeBaseIndexer
from kb_service.parser import parse_document

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# --- Redis Client Initialization ---
redis_client = redis.Redis(host=os.getenv("REDIS_HOST", "redis"), port=6379, db=0, decode_responses=True)

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

# --- Dynamic Controller Client Initialization ---
CONTROLLER_PROVIDER = os.getenv("CONTROLLER_PROVIDER", "openai").lower()
CONTROLLER_API_KEY = None
CONTROLLER_BASE_URL = None

if CONTROLLER_PROVIDER == "openrouter":
    CONTROLLER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    CONTROLLER_BASE_URL = "https://openrouter.ai/api/v1"
    logger.info("Configuring Controller to use OpenRouter.")
else: # Default to openai
    CONTROLLER_API_KEY = os.getenv("OPENAI_API_KEY")
    CONTROLLER_BASE_URL = "https://api.openai.com/v1"
    logger.info("Configuring Controller to use OpenAI.")

if not CONTROLLER_API_KEY:
    logger.warning("Controller API key is not set. The Quality Control stage will be skipped.")
    controller_client = None
else:
    controller_client = AsyncOpenAI(
        base_url=CONTROLLER_BASE_URL,
        api_key=CONTROLLER_API_KEY,
    )

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

def list_all_files_summary() -> str:
    """
    Lists a summary of all available files in the knowledge base.
    Returns a string with the name and ID of each file, useful for discovery.
    """
    logger.info("TOOL CALL: list_all_files_summary")
    try:
        all_files = kb_indexer.get_all_files()
        if not all_files:
            return "В базе знаний нет доступных файлов."
        
        summary = "Доступные файлы в базе знаний:\n"
        for f in all_files:
            summary += f"- Имя файла: '{f.get('name', 'N/A')}', ID: '{f.get('id', 'N/A')}'\n"
        return summary.strip()
    except Exception as e:
        logger.error(f"Error in list_all_files_summary tool: {e}", exc_info=True)
        return f"ОШИБКА: Не удалось получить список файлов: {e}"


# A robust retry decorator for all Google API calls
@retry(
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type((ResourceExhausted, InternalServerError))
)
async def run_with_retry(func, *args, **kwargs):
    """Executes an async function with a retry policy for specific API errors."""
    return await func(*args, **kwargs)


async def determine_file_context(user_message: str, all_files: List[Dict]) -> Optional[str]:
    """
    Analyzes the user's message to determine if it refers to a specific file.
    Returns the file_id if a match is found, otherwise None.
    """
    if not all_files:
        return None

    files_summary = "\n".join([f"- Имя файла: '{f.get('name', 'N/A')}', ID: '{f.get('id', 'N/A')}'" for f in all_files])
    
    prompt = f"""You are a classification assistant. Your task is to determine if the user's query refers to a specific file from the provided list.

Here is the list of available files:
<file_list>
{files_summary}
</file_list>

Here is the user's query:
<user_query>
{user_message}
</user_query>

Analyze the user's query. If it explicitly or implicitly refers to one of the files from the list, respond with ONLY the file's ID (e.g., "1a2b3c4d-5e6f-7a8b-9c0d-1e2f3a4b5c6d").
If the query does not refer to any specific file, respond with the exact word "None". Do not provide any other text or explanation.
"""
    try:
        context_model = genai.GenerativeModel('gemini-2.5-flash')
        # Use the retry helper for the API call
        response = await run_with_retry(context_model.generate_content_async, prompt)
        
        file_id_match = response.text.strip()
        
        # Validate that the returned ID is one of the available file IDs
        available_ids = {f.get('id') for f in all_files}
        if file_id_match in available_ids:
            logger.info(f"Context analysis determined the query refers to file_id: {file_id_match}")
            return file_id_match
        else:
            logger.info("Context analysis did not find a specific file reference.")
            return None
    except Exception as e:
        logger.error(f"Error during context determination: {e}")
        return None


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
CONFIG_FILE = "/app_config/config.json"
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
    use_agent_mode: bool = False

class CreateChatRequest(BaseModel):
    title: str

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

class JobCreationResponse(BaseModel):
    job_id: str


# --- API Endpoints ---
def load_config() -> AppConfig:
    default_config = AppConfig(
        executor=AgentSettings(model_name='gemini-2.5-pro', system_prompt='You are a helpful assistant.'),
        controller=AgentSettings(model_name='o4-mini', system_prompt='You are a helpful assistant.')
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
            os.remove(CONFIG_FILE)
        except OSError as del_e:
            logger.error(f"Failed to delete corrupt config file: {del_e}")
        return default_config

def save_config(config: AppConfig):
    # Ensure the directory exists before writing
    os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
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

@app.get("/api/v1/chats", response_model=List[ChatInfo])
async def list_chats():
    chats = []
    os.makedirs(HISTORY_DIR, exist_ok=True)
    for filename in os.listdir(HISTORY_DIR):
        if filename.endswith(".json"):
            conversation_id = filename[:-5]
            title_path = os.path.join(HISTORY_DIR, f"{conversation_id}.title.txt")
            title = "Новый чат"
            if os.path.exists(title_path):
                with open(title_path, 'r', encoding='utf-8') as f:
                    title = f.read().strip() or title
            else:
                try:
                    with open(os.path.join(HISTORY_DIR, filename), 'r', encoding='utf-8') as f:
                        history = json.load(f)
                        if history:
                            first_user_message = next((item for item in history if item.get('role') == 'user'), None)
                            if first_user_message and first_user_message.get('parts'):
                               title = first_user_message['parts'][0][:50]
                except (json.JSONDecodeError, IndexError) as e:
                    logger.warning(f"Could not generate title for {filename} due to error: {e}")
                    pass
            chats.append(ChatInfo(id=conversation_id, title=title))
    return sorted(chats, key=lambda item: os.path.getmtime(os.path.join(HISTORY_DIR, f"{item.id}.json")), reverse=True)

@app.post("/api/v1/chats", response_model=ChatInfo, status_code=status.HTTP_201_CREATED)
async def create_new_chat(request: CreateChatRequest):
    conversation_id = str(uuid.uuid4())
    history_file_path = os.path.join(HISTORY_DIR, f"{conversation_id}.json")
    title_file_path = os.path.join(HISTORY_DIR, f"{conversation_id}.title.txt")
    try:
        os.makedirs(HISTORY_DIR, exist_ok=True)
        with open(history_file_path, 'w', encoding='utf-8') as f: json.dump([], f)
        with open(title_file_path, 'w', encoding='utf-8') as f: f.write(request.title)
        return ChatInfo(id=conversation_id, title=request.title)
    except OSError as e:
        raise HTTPException(status_code=500, detail="Failed to create chat files.")

@app.post("/api/v1/jobs", response_model=JobCreationResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_chat_job(request: ChatRequest):
    job_id = f"job:{uuid.uuid4()}"
    job_data = request.model_dump_json()

    # --- SAVE USER MESSAGE IMMEDIATELY ---
    try:
        history_file_path = os.path.join(HISTORY_DIR, f"{request.conversation_id}.json")
        if not os.path.exists(history_file_path):
            # This should not happen if the chat was created correctly, but as a safeguard:
            with open(history_file_path, 'w', encoding='utf-8') as f:
                json.dump([], f)

        with open(history_file_path, 'r+', encoding='utf-8') as f:
            try:
                # Read existing history
                history = json.load(f)
                if not isinstance(history, list): history = []
            except json.JSONDecodeError:
                history = [] # Overwrite if file is corrupt
            
            # Append ONLY the new user message
            history.append({"role": "user", "parts": [request.message]})
            
            # Go back to the beginning of the file to overwrite
            f.seek(0)
            json.dump(history, f, indent=2, ensure_ascii=False)
            f.truncate()
    except Exception as e:
        logger.error(f"Failed to write user message to history file {history_file_path}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to save user message.")

    # --- LINK CONVERSATION TO JOB ---
    redis_client.set(f"active_job_for_convo:{request.conversation_id}", job_id, ex=3600) # ex=3600 sets expiry to 1 hour
    logger.info(f"Linked conversation {request.conversation_id} to active job {job_id}")

    # Create initial status in a Redis Hash
    initial_status = {
        "status": "queued",
        "thoughts": json.dumps([{"type": "info", "content": "Задача поставлена в очередь..."}]),
        "final_answer": ""
    }
    
    redis_client.hset(job_id, mapping=initial_status)
    
    # Push the job details to the worker queue
    redis_client.lpush("job_queue", json.dumps({"job_id": job_id, "payload": job_data}))
    
    logger.info(f"Job {job_id} created and queued for conversation {request.conversation_id}.")
    return JobCreationResponse(job_id=job_id)

@app.get("/api/v1/jobs/{job_id}/status")
async def get_job_status(job_id: str):
    job_data = redis_client.hgetall(job_id)
    if not job_data:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Deserialize the thoughts list
    job_data['thoughts'] = json.loads(job_data.get('thoughts', '[]'))
    return JSONResponse(content=job_data)

@app.post("/api/v1/jobs/{job_id}/cancel", status_code=status.HTTP_200_OK)
async def cancel_job(job_id: str):
    if not redis_client.exists(job_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    
    redis_client.hset(job_id, "status", "cancelled")
    
    logger.info(f"Job {job_id} cancellation request received and status set to 'cancelled'.")
    return JSONResponse(content={"status": "cancellation_requested", "job_id": job_id})

@app.get("/api/v1/chats/{conversation_id}/active_job", status_code=200)
async def get_active_job_for_convo(conversation_id: str):
    job_id_key = f"active_job_for_convo:{conversation_id}"
    job_id = redis_client.get(job_id_key)
    
    if not job_id:
        return {"job_id": None}

    # Also check if the job itself still exists in Redis
    if not redis_client.exists(job_id):
         # The link is stale, clean it up
         redis_client.delete(job_id_key)
         return {"job_id": None}
         
    return {"job_id": job_id}

@app.get("/api/v1/chats/{conversation_id}")
async def get_chat_history(conversation_id: str):
    history_file_path = os.path.join(HISTORY_DIR, f"{conversation_id}.json")
    if not os.path.exists(history_file_path):
        raise HTTPException(status_code=404, detail="Chat history not found.")
    
    try:
        with open(history_file_path, 'r', encoding='utf-8') as f:
            # First, read the content to see if the file is empty
            content = f.read()
            if not content.strip():
                return [] # Return empty list for empty or whitespace-only files
            # If not empty, try to parse
            history_data = json.loads(content)
        
        formatted_history = []
        for item in history_data:
            # Robustly get content from parts
            parts = item.get("parts", [])
            content_text = parts[0] if parts else ""

            # Ensure 'sources' key is always present, defaulting to an empty list
            sources = item.get("sources", [])

            message_data = {
                "role": item.get("role"), 
                "content": content_text,
                "sources": sources # This line is critical
            }

            if 'thinking_steps' in item and item['thinking_steps']:
                message_data['thinking_steps'] = item['thinking_steps']

            formatted_history.append(message_data)
        return formatted_history
        
    except json.JSONDecodeError:
        logger.warning(f"Could not parse corrupted chat history file for conversation_id: {conversation_id}. Returning empty history.")
        return []
    except Exception as e:
        logger.error(f"An unexpected error occurred while reading history for {conversation_id}: {e}", exc_info=True)
        # For any other unexpected error, also return an empty list to prevent frontend crash
        return []

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
