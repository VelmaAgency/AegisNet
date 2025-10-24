import logging

logger = logging.getLogger(__name__)

class BillingAPI:
    async def create_customer(self, email):
        logger.info("Customer created", extra={"email": email})
