import os
import time
import logging
import json
import redis
import google.generativeai as genai
import google.generativeai.protos as gap
from openai import AsyncOpenAI
import asyncio
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
from google.api_core.exceptions import ResourceExhausted, InternalServerError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# --- Setup Logging and Environment ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
load_dotenv()

# --- Load all necessary components from kb_service ---
# The docker-compose volume mount makes this possible: `...:/app/kb_service`
from kb_service.connector import MockConnector
from kb_service.yandex_connector import YandexDiskConnector
from kb_service.indexer import KnowledgeBaseIndexer

# --- Pydantic Models ---
class AgentSettings(BaseModel):
    model_name: str
    system_prompt: str

class AppConfig(BaseModel):
    executor: AgentSettings
    controller: AgentSettings

class ThinkingStep(BaseModel):
    type: str
    content: str

class Message(BaseModel):
    role: str
    parts: List[str]
    thinking_steps: Optional[List[ThinkingStep]] = None

# --- Constants ---
HISTORY_DIR = "/app/chat_histories"
CONFIG_FILE = "/app_config/config.json"
CONTROLLER_SYSTEM_PROMPT = "You are a helpful assistant."

os.makedirs(HISTORY_DIR, exist_ok=True)


# --- Redis Helper Functions ---
def update_job_status(r_client: redis.Redis, job_id: str, new_thought: str = None, final_answer: str = None, status: str = None):
    try:
        update_data = {}
        
        # Always handle thoughts update, it's the main purpose of this function during processing
        if new_thought:
            current_thoughts_raw = r_client.hget(job_id, "thoughts")
            current_thoughts = json.loads(current_thoughts_raw) if current_thoughts_raw else []
            current_thoughts.append({"type": "thought", "content": new_thought})
            update_data["thoughts"] = json.dumps(current_thoughts)

        # Check current status first to prevent overwriting a terminal state
        current_job_status = r_client.hget(job_id, "status")
        is_terminal = current_job_status in ["complete", "failed", "cancelled"]

        if is_terminal:
            logger.warning(f"Job {job_id} is in a terminal state ({current_job_status}). Only updating thoughts.")
        else:
            # Only add final_answer and status if the job is not in a terminal state
            if final_answer is not None:
                update_data["final_answer"] = final_answer
            if status is not None:
                update_data["status"] = status

        # If nothing is left to update, exit.
        if not update_data:
            return
            
        r_client.hset(job_id, mapping=update_data)

        log_status = status if not is_terminal else f"(ignored, state is {current_job_status})"
        logger.info(f"Updated job {job_id}: new_thought='{new_thought}', status='{log_status}'")

    except Exception as e:
        logger.error(f"Failed to update status for job {job_id}: {e}")

# --- Config Helper ---
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
        logger.warning(f"Could not load config file due to: {e}. Using defaults.")
        return default_config

# --- AI Client Initializations ---
logger.info("Initializing AI clients and services...")
# Knowledge Base
YANDEX_TOKEN = os.getenv("YANDEX_DISK_API_TOKEN")
kb_connector = YandexDiskConnector(token=YANDEX_TOKEN) if YANDEX_TOKEN else MockConnector()
kb_indexer = KnowledgeBaseIndexer(connector=kb_connector)
kb_indexer.build_index()  # Build index on worker startup

# Gemini (Executor)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    raise ValueError("GEMINI_API_KEY environment variable not set!")

# OpenAI/OpenRouter (Controller)
CONTROLLER_PROVIDER = os.getenv("CONTROLLER_PROVIDER", "openai").lower()
CONTROLLER_API_KEY = os.getenv("OPENROUTER_API_KEY") if CONTROLLER_PROVIDER == "openrouter" else os.getenv("OPENAI_API_KEY")
CONTROLLER_BASE_URL = "https://openrouter.ai/api/v1" if CONTROLLER_PROVIDER == "openrouter" else "https://api.openai.com/v1"
controller_client = AsyncOpenAI(base_url=CONTROLLER_BASE_URL, api_key=CONTROLLER_API_KEY) if CONTROLLER_API_KEY else None
if controller_client:
    logger.info(f"Controller configured to use {CONTROLLER_PROVIDER}.")
else:
    logger.warning("Controller client not configured. Quality Control will be skipped.")

# --- AI Tools Definition ---
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
    logger.info("TOOL CALL: list_all_files_summary")
    try:
        all_files = kb_indexer.get_all_files()
        if not all_files: return "В базе знаний нет доступных файлов."
        summary = "Доступные файлы в базе знаний:\n" + "\n".join([f"- Имя файла: '{f.get('name', 'N/A')}', ID: '{f.get('id', 'N/A')}'" for f in all_files])
        return summary.strip()
    except Exception as e:
        logger.error(f"Error in list_all_files_summary tool: {e}", exc_info=True)
        return f"ОШИБКА: Не удалось получить список файлов: {e}"

# --- AI Helper Functions ---
@retry(
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type((ResourceExhausted, InternalServerError))
)
async def run_with_retry(func, *args, **kwargs):
    return await func(*args, **kwargs)

async def determine_file_context(user_message: str, all_files: List[Dict]) -> Optional[str]:
    if not all_files: return None
    files_summary = "\n".join([f"- Имя файла: '{f.get('name', 'N/A')}', ID: '{f.get('id', 'N/A')}'" for f in all_files])
    prompt = f"You are a classification assistant... (full prompt from original file)... If it refers to one of the files from the list, respond with ONLY the file's ID... If not, respond with 'None'."
    try:
        context_model = genai.GenerativeModel('gemini-1.5-flash')
        response = await run_with_retry(context_model.generate_content_async, prompt)
        file_id_match = response.text.strip()
        if file_id_match in {f.get('id') for f in all_files}:
            logger.info(f"Context analysis determined the query refers to file_id: {file_id_match}")
            return file_id_match
        return None
    except Exception as e:
        logger.error(f"Error during context determination: {e}")
        return None

# --- AI Core Logic ---
async def process_ai_task(job_id: str, request_payload: dict, r_client: redis.Redis):
    try:
        config = load_config()

        if r_client.hget(job_id, "status") == "cancelled":
            logger.info(f"Job {job_id} was cancelled. Aborting before processing.")
            return
            
        MAX_ITERATIONS = 3
        feedback_from_controller = ""
        final_approved_answer = "Агент не смог сформировать ответ."
        tool_context = ""
        request_message = request_payload['message']
        request_file_id = request_payload.get('file_id')
        conversation_id = request_payload['conversation_id']

        for iteration in range(MAX_ITERATIONS):
            if r_client.hget(job_id, "status") == "cancelled":
                logger.info(f"Job {job_id} was cancelled during iteration {iteration + 1}. Aborting.")
                return

            # STAGE 1: EXECUTOR
            step_content = f"[Исполнитель][Итерация {iteration+1}/{MAX_ITERATIONS}] "
            step_content += "Получены правки от контроля качества. Начинаю доработку..." if iteration > 0 else "Анализирую запрос и готовлю ответ..."
            update_job_status(r_client, job_id, new_thought=step_content)

            if iteration == 0 and not request_file_id:
                update_job_status(r_client, job_id, new_thought="[Анализ] Определяю, относится ли запрос к файлу...")
                all_files = kb_indexer.get_all_files()
                contextual_file_id = await determine_file_context(request_message, all_files)
                if contextual_file_id:
                    request_file_id = contextual_file_id
                    update_job_status(r_client, job_id, new_thought=f"[Анализ] Контекст определен. Работа с файлом ID: {contextual_file_id}")

            prompt_for_executor = request_message
            if iteration == 0 and request_file_id:
                prompt_for_executor += f"\n\n[Контекст определен] Работай с файлом ID: {request_file_id}."
            elif iteration > 0:
                prompt_for_executor = f"Your previous answer requires changes. Feedback: {feedback_from_controller}. Original query: {request_message}"
            
            model = genai.GenerativeModel(
                model_name=config.executor.model_name,
                tools=[analyze_document, search_knowledge_base, list_all_files_summary],
                system_instruction=config.executor.system_prompt
            )
            chat_session = model.start_chat(history=[])
            response = await run_with_retry(chat_session.send_message_async, prompt_for_executor)
            executor_answer = "Исполнитель не смог сформировать ответ."
            tool_context = ""

            while True:
                if r_client.hget(job_id, "status") == "cancelled":
                    logger.info(f"Job {job_id} was cancelled during tool use. Aborting.")
                    return
                if not response.candidates: break
                part = response.candidates[0].content.parts[0]
                if hasattr(part, 'function_call') and part.function_call.name:
                    fc = part.function_call
                    tool_map = {"analyze_document": analyze_document, "search_knowledge_base": search_knowledge_base, "list_all_files_summary": list_all_files_summary}
                    tool_func = tool_map.get(fc.name)
                    tool_result = tool_func(**fc.args) if tool_func else f"Ошибка: Неизвестный инструмент '{fc.name}'."
                    tool_context += f"Вызов {fc.name} с {fc.args} дал результат:\n{tool_result}\n\n"
                    update_job_status(r_client, job_id, new_thought=f"Использую инструмент: {fc.name}({fc.args})")
                    response = await run_with_retry(chat_session.send_message_async, gap.Part(function_response=gap.FunctionResponse(name=fc.name, response={'content': tool_result})))
                elif hasattr(part, 'text') and part.text:
                    executor_answer = part.text
                    break
                else: break
            
            final_approved_answer = executor_answer

            # STAGE 2: CONTROLLER
            if not controller_client:
                update_job_status(r_client, job_id, new_thought="Контроль качества пропущен (не настроен).")
                break
            
            if r_client.hget(job_id, "status") == "cancelled":
                logger.info(f"Job {job_id} was cancelled before controller. Aborting.")
                return

            update_job_status(r_client, job_id, new_thought=f"[Контроль][Итерация {iteration+1}] Проверяю качество...")
            controller_prompt = f"User query: <user_query>{request_message}</user_query>\nRetrieved context: <retrieved_context>{tool_context or 'None'}</retrieved_context>\nAnswer to review: <answer_to_review>{executor_answer}</answer_to_review>\nIs the answer complete and accurate? Respond with JSON: {{'is_approved': boolean, 'feedback': string}}."
            controller_model_name = os.getenv("CONTROLLER_MODEL_NAME") or config.controller.model_name
            controller_response = await controller_client.chat.completions.create(model=controller_model_name, messages=[{"role": "system", "content": config.controller.system_prompt}, {"role": "user", "content": controller_prompt}], response_format={"type": "json_object"})
            review_data = json.loads(controller_response.choices[0].message.content)

            if review_data.get("is_approved"):
                update_job_status(r_client, job_id, new_thought=f"[Контроль][Итерация {iteration+1}] Качество подтверждено.")
                break
            else:
                feedback_from_controller = review_data.get("feedback", "Требуются улучшения.")
                update_job_status(r_client, job_id, new_thought=f"[Контроль][Итерация {iteration+1}] Обнаружены недочеты: {feedback_from_controller}")
                if iteration == MAX_ITERATIONS - 1:
                    logger.warning("Max iterations reached for job {job_id}. Using the last answer.")

        # FINALIZATION STAGE
        update_job_status(r_client, job_id, final_answer=final_approved_answer, status="complete")
        
        # Save chat history
        history_file_path = os.path.join(HISTORY_DIR, f"{conversation_id}.json")
        history = []
        if os.path.exists(history_file_path):
            with open(history_file_path, 'r', encoding='utf-8') as f:
                history = json.load(f)
        user_message = Message(role="user", parts=[request_message])
        
        # Fetch final thoughts from Redis as the single source of truth
        final_thoughts_raw = r_client.hget(job_id, "thoughts")
        final_thinking_steps = json.loads(final_thoughts_raw) if final_thoughts_raw else []

        # Now create the model message with the complete data
        model_message = Message(role="model", parts=[final_approved_answer], thinking_steps=[ThinkingStep(**step) for step in final_thinking_steps])

        history.extend([user_message.model_dump(exclude_none=True), model_message.model_dump(exclude_none=True)])
        
        # Ensure directory exists before writing history
        os.makedirs(os.path.dirname(history_file_path), exist_ok=True)
        with open(history_file_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Job {job_id} finished successfully and history saved.")

    except Exception as e:
        logger.error(f"Critical error during AI task for job {job_id}: {e}", exc_info=True)
        update_job_status(r_client, job_id, new_thought=f"Критическая ошибка: {e}", status="failed")

# --- Main Worker Loop ---
async def main_worker_loop():
    logger.info("AI Worker is running and waiting for tasks.")
    redis_client = redis.Redis(host=os.getenv("REDIS_HOST", "redis"), port=6379, db=0, decode_responses=True)

    while True:
        try:
            _, job_raw = redis_client.brpop("job_queue", timeout=0)
            job_data = json.loads(job_raw)
            job_id = job_data.get("job_id")
            payload = json.loads(job_data.get("payload", "{}"))

            if not job_id:
                logger.warning("Skipping job with no job_id.")
                continue

            logger.info(f"Picked up job: {job_id}")
            update_job_status(redis_client, job_id, new_thought="Задача получена, начинаю обработку.", status="processing")

            await process_ai_task(job_id, payload, redis_client)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode job from Redis: {e}. Raw data: '{job_raw}'")
        except Exception as e:
            logger.error(f"An error occurred in the main worker loop: {e}", exc_info=True)
            time.sleep(5)

if __name__ == "__main__":
    asyncio.run(main_worker_loop())
