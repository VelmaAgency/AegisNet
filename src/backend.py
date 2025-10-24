import logging
import asyncio
from typing import Dict

logger = logging.getLogger(__name__)

class Backend:
    def __init__(self, config: Dict):
        self.config = config
        logger.info("Backend initialized for customer management.")

    async def create_customer(self, email: str) -> None:
        try:
            logger.info("Customer created", extra={"email": email})
        except Exception as e:
            logger.error("Customer creation error", extra={"error": str(e)})
