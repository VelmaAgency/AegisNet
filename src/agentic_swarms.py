import logging
import asyncio
import uuid

logger = logging.getLogger(__name__)

class Agent:
    def __init__(self, role):
        self.role = role
        self.id = str(uuid.uuid4())

class AgenticSwarms:
    def __init__(self):
        self.agents = []

    async def spawn_hunters(self, count):
        max_agents = min(count, 50)
        for i in range(max_agents):
            self.agents.append(Agent(f"hunter_{i}"))
            logger.info("Spawned agent", extra={"agent_id": self.agents[-1].id})
