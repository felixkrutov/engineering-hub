import os
import time
import logging
import redis
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Worker started.")
    redis_host = os.getenv("REDIS_HOST", "redis")
    
    try:
        r = redis.Redis(host=redis_host, port=6379, db=0, decode_responses=True)
        r.ping()
        logger.info("Successfully connected to Redis.")
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}", exc_info=True)
        return

    logger.info("Waiting for tasks from 'job_queue'...")
    while True:
        try:
            # BRPOP is a blocking call that waits for an item on the list
            # It waits indefinitely (timeout=0)
            _ , job_raw = r.brpop("job_queue", timeout=0)
            
            job_data = json.loads(job_raw)
            job_id = job_data.get("job_id")
            
            if not job_id:
                logger.warning("Received a job with no job_id. Skipping.")
                continue
            
            logger.info(f"Picked up job: {job_id}")

            # Update job status to 'processing'
            r.hset(job_id, "status", "processing")
            r.hset(job_id, "thoughts", json.dumps([{"type": "info", "content": "Задача в обработке..."}]))
            
            # --- THIS IS WHERE THE AI LOGIC WILL GO IN THE NEXT STEP ---
            # For now, we simulate work and set it to 'complete'
            time.sleep(5) # Simulate work
            
            # Update status to 'complete' and set a mock answer
            final_answer = f"Job {job_id} processed successfully by the worker."
            final_status = {
                "status": "complete",
                "final_answer": final_answer,
                "thoughts": json.dumps([
                    {"type": "info", "content": "Задача в обработке..."},
                    {"type": "info", "content": "Работа завершена."}
                ]),
            }
            r.hset(job_id, mapping=final_status)
            logger.info(f"Job {job_id} finished.")

        except Exception as e:
            logger.error(f"An error occurred during job processing: {e}", exc_info=True)
            # Avoid crashing the worker on a single job failure
            time.sleep(5)

if __name__ == "__main__":
    main()
