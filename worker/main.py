# worker/main.py

import os
import time
import logging
import redis

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
        # Exit if we can't connect to Redis, as the worker is useless.
        return

    logger.info("Waiting for tasks...")
    # In the future, this will be a loop that pulls tasks from Redis.
    while True:
        # logger.info("Worker is alive and waiting...")
        time.sleep(60)

if __name__ == "__main__":
    main()
