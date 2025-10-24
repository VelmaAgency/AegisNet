import logging
import asyncio
import aiohttp
from typing import Dict

logger = logging.getLogger(__name__)

class UemManager:
    def __init__(self, config: Dict):
        self.api_endpoint = config['suremdm_api_endpoint']
        self.api_key = config['suremdm_api_key']
        self.session = aiohttp.ClientSession(headers={"Authorization": f"Bearer {self.api_key}"})
        logger.info("UEM Manager initialized for SureMDM/Intune.")

    async def manage_device(self, device_id: str, action: str, params: Dict = None) -> Dict:
        try:
            if action == "mfa_enable":
                endpoint = f"{self.api_endpoint}/devices/{device_id}/mfa"
                response = await self.session.post(endpoint, json=params)
            elif action == "whitelist_app":
                endpoint = f"{self.api_endpoint}/devices/{device_id}/apps/whitelist"
                response = await self.session.post(endpoint, json=params)
            elif action == "patch":
                endpoint = f"{self.api_endpoint}/devices/{device_id}/patch"
                response = await self.session.post(endpoint, json=params)
            else:
                raise ValueError(f"Unsupported action: {action}")

            if response.status == 200:
                logger.info(f"UEM action {action} succeeded", extra={"device_id": device_id})
                return {"status": "success", "details": await response.json()}
            else:
                logger.error(f"UEM action {action} failed", extra={"device_id": device_id, "status": response.status})
                return {"status": "error", "details": await response.text()}
        except Exception as e:
            logger.error("UEM action error", extra={"error": str(e)})
            return {"status": "error", "details": str(e)}
        finally:
            await self.session.close()
