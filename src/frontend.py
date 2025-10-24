import logging

logger = logging.getLogger(__name__)

class DemoPortal:
    async def configure_grafana_dashboard(self, dashboard_id):
        logger.info("Grafana dashboard configured", extra={"id": dashboard_id})
