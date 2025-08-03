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
from openai import AsyncOpenAI

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
            os.remove(CONFIG_FILE)
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


@app.post("/api/v1/chat/stream")
async def stream_chat(request: ChatRequest) -> StreamingResponse:
    async def event_generator():
        steps_history: List[Dict[str, Any]] = []
        final_answer_text = "Произошла ошибка при обработке ответа."

        conversation_id = request.conversation_id
        os.makedirs(HISTORY_DIR, exist_ok=True)
        history_file_path = os.path.join(HISTORY_DIR, f"{conversation_id}.json")

        history = []
        if not os.path.exists(history_file_path):
            with open(history_file_path, 'w', encoding='utf-8') as f:
                json.dump([], f)
        
        try:
            with open(history_file_path, 'r', encoding='utf-8') as f:
                history = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            logger.warning(f"Could not read or parse history for {conversation_id}, starting fresh.")
            history = []
        
        try:
            config = load_config()
            MAX_ITERATIONS = 3
            feedback_from_controller = ""
            final_approved_answer = "Агент не смог сформировать ответ."
            tool_context = ""
            
            for iteration in range(MAX_ITERATIONS):
                # --- STAGE 1: EXECUTOR (AI1) ---
                step_content = f"[Исполнитель][Итерация {iteration+1}/{MAX_ITERATIONS}] "
                if iteration > 0:
                    step_content += f"Получены правки от контроля качества. Начинаю доработку..."
                else:
                    step_content += "Анализирую запрос и готовлю ответ..."
                
                steps_history.append({'type': 'thought', 'content': step_content})
                yield f"data: {json.dumps(steps_history[-1])}\n\n"

                prompt_for_executor = request.message
                if iteration > 0:
                    prompt_for_executor = f"""Your previous answer was reviewed and requires changes. Please generate a new, improved final-quality response that addresses the following feedback.
<feedback>
{feedback_from_controller}
</feedback>

The original user query was:
<original_query>
{request.message}
</original_query>
"""
                
                model = genai.GenerativeModel(
                    model_name=config.executor.model_name,
                    tools=[analyze_document, search_knowledge_base, list_all_files_summary],
                    system_instruction=config.executor.system_prompt
                )
                chat_session = model.start_chat(history=[]) # Start fresh for each iteration
                
                response = await chat_session.send_message_async(prompt_for_executor)
                executor_answer = "Исполнитель не смог сформировать ответ на данной итерации."
                tool_context = "" # Reset context for this iteration

                # --- Executor's internal tool-use loop ---
                while True:
                    if not response.candidates:
                        executor_answer = "Модель-исполнитель не вернула кандидатов."
                        break
                    
                    part = response.candidates[0].content.parts[0]
                    
                    if hasattr(part, 'function_call') and part.function_call.name:
                        fc = part.function_call
                        tool_map = {
                            "analyze_document": analyze_document,
                            "search_knowledge_base": search_knowledge_base,
                            "list_all_files_summary": list_all_files_summary,
                        }
                        
                        tool_func = tool_map.get(fc.name)
                        if not tool_func:
                            tool_result = f"Ошибка: Неизвестный инструмент '{fc.name}'."
                        else:
                            # Safely call the function with its arguments
                            tool_result = tool_func(**fc.args)

                        tool_context += f"Вызов инструмента {fc.name} с аргументами {fc.args} дал результат:\n{tool_result}\n\n"

                        response = await chat_session.send_message_async(
                            gap.Part(function_response=gap.FunctionResponse(name=fc.name, response={'content': tool_result}))
                        )
                    elif hasattr(part, 'text') and part.text:
                        executor_answer = part.text
                        break
                    else:
                        executor_answer = "Модель-исполнитель вернула пустой ответ."
                        break
                
                final_approved_answer = executor_answer

                # --- STAGE 2: CONTROLLER (AI2) ---
                if not controller_client:
                    logger.info("Controller client not configured. Approving answer by default.")
                    break

                step_data = {'type': 'thought', 'content': f"[Контроль][Итерация {iteration+1}] Проверяю качество ответа..."}
                steps_history.append(step_data)
                yield f"data: {json.dumps(step_data)}\n\n"

                controller_prompt = f"""You are a Quality Control auditor. Your task is to review the response generated by an Engineer AI.
The original user query was: <user_query>{request.message}</user_query>
The Engineer AI used its tools and retrieved this context: <retrieved_context>{tool_context if tool_context else "None"}</retrieved_context>
The Engineer AI produced this answer: <answer_to_review>{executor_answer}</answer_to_review>
Critically evaluate the answer. Is it complete, accurate, and fully addresses the user's query?
Your response MUST be a valid JSON object with two keys: "is_approved" (boolean) and "feedback" (string). If the answer is perfect, set "is_approved" to true. If it needs any improvement, set it to false and provide concise, actionable feedback.
"""
                
                controller_model = os.getenv("CONTROLLER_MODEL_NAME") or config.controller.model_name
                controller_response = await controller_client.chat.completions.create(
                    model=controller_model,
                    messages=[{"role": "system", "content": config.controller.system_prompt}, {"role": "user", "content": controller_prompt}],
                    response_format={"type": "json_object"}
                )
                review_data = json.loads(controller_response.choices[0].message.content)
                
                if review_data.get("is_approved"):
                    step_data = {'type': 'thought', 'content': f"[Контроль][Итерация {iteration+1}] Качество подтверждено."}
                    steps_history.append(step_data); yield f"data: {json.dumps(step_data)}\n\n"
                    break # EXIT THE LOOP
                else:
                    feedback_from_controller = review_data.get("feedback", "Необходимо внести улучшения.")
                    step_data = {'type': 'thought', 'content': f"[Контроль][Итерация {iteration+1}] Обнаружены недочеты. Комментарий: {feedback_from_controller}"}
                    steps_history.append(step_data); yield f"data: {json.dumps(step_data)}\n\n"
                    if iteration == MAX_ITERATIONS - 1:
                        logger.warning("Max iterations reached. Using the last available answer.")

        except Exception as e:
            logger.error(f"Critical error in Quality Control Loop: {e}", exc_info=True)
            final_approved_answer = f"Критическая ошибка в цикле обработки: {e}"

        # --- FINALIZATION STAGE ---
        final_answer_text = final_approved_answer
        
        user_message = Message(role="user", parts=[request.message])
        model_message = Message(role="model", parts=[final_answer_text], thinking_steps=[ThinkingStep(**step) for step in steps_history])
        
        history.append(user_message.model_dump(exclude_none=True))
        history.append(model_message.model_dump(exclude_none=True))

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
