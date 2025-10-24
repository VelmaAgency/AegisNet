import logging
import aiohttp
import asyncio

logger = logging.getLogger(__name__)

class ComplianceSuite:
    async def fetch_regulatory_feed(self, regulation):
        async with aiohttp.ClientSession() as session:
            response = await session.get(f"https://chain.link/api/{regulation}")
            data = await response.json()
            logger.info("Regulatory feed fetched", extra={"regulation": regulation})
            return data
