import logging

logger = logging.getLogger(__name__)

class IAMScoper:
    async def scope_iam(self, user, action):
        if action == "provision_device":
            logger.info("IAM scoped for Jamf", extra={"user_id": user["id"]})
            return True
