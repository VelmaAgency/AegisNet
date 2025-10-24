import logging
from scapy.all import AsyncSniffer

logger = logging.getLogger(__name__)

class WirelessSensorGuard:
    def __init__(self):
        self.sniffer = AsyncSniffer(filter="wlan type mgt subtype probe-req")

    async def detect_rogue_ap(self, packet):
        if packet.haslayer(Dot11ProbeReq) and packet.ssid not in ["AegisNet"]:
            logger.warning("Rogue AP detected", extra={"ssid": packet.ssid, "bssid": packet.addr2})
            return True
        return False
