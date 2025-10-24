import logging

logger = logging.getLogger(__name__)

class AntMesh:
    async def route_packet(self, packet):
        logger.info("Packet routed", extra={"node_id": packet["node_id"]})
        return True
