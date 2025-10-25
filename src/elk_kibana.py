# elk_kibana.py - Kibana Visualization for ELK Logs
import requests
import json
import logging
from typing import Dict
import structlog

structlog.configure(processors=[structlog.processors.JSONRenderer()])
logger = structlog.get_logger(__name__)

class KibanaVisualizer:
    """Visualize ELK logs in Kibana for AegisNet v2.1.1."""
    def __init__(self, kibana_url: str = "http://localhost:5601", elk_index: str = "aegisnet-transfer-2025.10.24"):
        self.kibana_url = kibana_url
        self.elk_index = elk_index

    def create_kibana_dashboard(self) -> Dict:
        """Generate Kibana dashboard JSON for ELK logs."""
        dashboard = {
            "attributes": {
                "title": "AegisNet v2.1.1 Threat Logs",
                "visualizationList": [
                    {
                        "type": "line",
                        "title": "YARA Match Trends",
                        "query": {
                            "query": "matches:*",
                            "language": "lucene"
                        },
                        "metrics": [{"aggregation": "count", "field": "matches.rule"}]
                    },
                    {
                        "type": "metric",
                        "title": "Error Rate",
                        "query": {
                            "query": "level:ERROR",
                            "language": "lucene"
                        },
                        "metrics": [{"aggregation": "count"}]
                    },
                    {
                        "type": "table",
                        "title": "Threat Details",
                        "query": {
                            "query": "matches:*",
                            "language": "lucene"
                        },
                        "columns": [{"field": "matches.rule"}, {"field": "timestamp"}]
                    }
                ]
            }
        }
        return dashboard

    def query_elk_logs(self, stix2_id: str = "transfer–v1.2-008") -> Dict:
        """Query ELK logs for Kibana visualization."""
        try:
            url = f"{self.kibana_url}/api/saved_objects/_find?type=index-pattern&search={self.elk_index}"
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            logger.info("ELK query successful", stix2_id=stix2_id)
            return response.json()
        except Exception as e:
            logger.error(f"ELK query error: {e}")
            return {"status": "error", "error": str(e)}

# Example usage
if __name__ == "__main__":
    visualizer = KibanaVisualizer()
    dashboard = visualizer.create_kibana_dashboard()
    print(json.dumps(dashboard, indent=2))
    print(visualizer.query_elk_logs())