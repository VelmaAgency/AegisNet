import logging

logger = logging.getLogger(__name__)

class DemoPortal:
    async def configure_grafana_dashboard(self, dashboard_id):
        logger.info("Grafana dashboard configured", extra={"id": dashboard_id})
# frontend.py - Grafana Dashboard Integration for AegisNet v2.1.1
from flask import Flask, Response
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from core import APTShield, BioTriad
import torch
import json
import logging
from typing import Dict

logger = logging.getLogger(__name__)
app = Flask(__name__)

# Prometheus metrics
yara_matches = Counter('aegisnet_yara_matches_total', 'Total YARA rule matches', ['rule'])
yara_latency = Histogram('aegisnet_yara_scan_latency_seconds', 'YARA scan latency')
recovery_success = Gauge('aegisnet_recovery_success', 'Bio-Triad recovery success rate')
anomaly_score = Gauge('aegisnet_anomaly_score', 'Anomaly detection score')

def setup_grafana_dashboard() -> Dict:
    """Generate Grafana dashboard JSON for metrics."""
    dashboard = {
        "dashboard": {
            "title": "AegisNet v2.1.1 Cybersecurity Dashboard",
            "panels": [
                {
                    "type": "graph",
                    "title": "YARA Rule Matches",
                    "targets": [
                        {
                            "expr": 'rate(aegisnet_yara_matches_total[5m])',
                            "legendFormat": "{{rule}}"
                        }
                    ],
                    "yaxes": [{"format": "short"}]
                },
                {
                    "type": "graph",
                    "title": "YARA Scan Latency",
                    "targets": [
                        {
                            "expr": 'aegisnet_yara_scan_latency_seconds',
                            "legendFormat": "Latency"
                        }
                    ],
                    "yaxes": [{"format": "s"}]
                },
                {
                    "type": "gauge",
                    "title": "Bio-Triad Recovery Rate",
                    "targets": [
                        {
                            "expr": 'aegisnet_recovery_success',
                            "legendFormat": "Recovery"
                        }
                    ],
                    "valueMappings": [{"value": 1, "text": "Success"}]
                },
                {
                    "type": "stat",
                    "title": "Anomaly Score",
                    "targets": [
                        {
                            "expr": 'aegisnet_anomaly_score',
                            "legendFormat": "Score"
                        }
                    ],
                    "thresholds": [{"value": 0.93, "color": "red"}]
                }
            ]
        },
        "overwrite": True
    }
    return dashboard

@app.route('/metrics')
def metrics():
    """Expose Prometheus metrics for Grafana."""
    try:
        # Mock data for demonstration
        shield = APTShield()
        triad = BioTriad()
        data = b"MOVEit.DMZ X-siLock-Comment"
        input_tensor = torch.rand(128)
        
        # Update metrics
        matches = shield.scan_data(data)
        for match in matches:
            yara_matches.labels(rule=match).inc()
        recovery_success.set(1 if triad.planarian_healing(input_tensor, 0.95, 1.0) else 0)
        anomaly_score.set(0.95)  # Mock score
        
        return Response(generate_latest(), mimetype='text/plain')
    except Exception as e:
        logger.error(f"Metrics error: {e}")
        return Response("Error", status=500)

@app.route('/dashboard')
def get_dashboard():
    """Return Grafana dashboard JSON."""
    try:
        dashboard = setup_grafana_dashboard()
        return Response(json.dumps(dashboard), mimetype='application/json')
    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        return Response("Error", status=500)

if __name__ == "__main__":
    # Start Prometheus server
    from prometheus_client import start_http_server
    start_http_server(8000)  # Prometheus scrapes here
    app.run(host='0.0.0.0', port=5000)
    # frontend.py - Grafana Dashboard with Alerting for AegisNet v2.1.1
from flask import Flask, Response
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from core import APTShield, BioTriad
import torch
import json
import logging
from typing import Dict

logger = logging.getLogger(__name__)
app = Flask(__name__)

# Prometheus metrics
yara_matches = Counter('aegisnet_yara_matches_total', 'Total YARA rule matches', ['rule'])
yara_latency = Histogram('aegisnet_yara_scan_latency_seconds', 'YARA scan latency')
recovery_success = Gauge('aegisnet_recovery_success', 'Bio-Triad recovery success rate')
anomaly_score = Gauge('aegisnet_anomaly_score', 'Anomaly detection score')
cpu_usage = Gauge('aegisnet_cpu_usage_percent', 'CPU usage')
memory_usage = Gauge('aegisnet_memory_usage_mb', 'Memory usage')

def setup_grafana_dashboard() -> Dict:
    """Generate detailed Grafana dashboard JSON with alerting."""
    dashboard = {
        "dashboard": {
            "title": "AegisNet v2.1.1 Cybersecurity Dashboard",
            "panels": [
                {
                    "type": "timeseries",
                    "title": "YARA Rule Matches by Threat",
                    "targets": [
                        {
                            "expr": 'rate(aegisnet_yara_matches_total{rule=~"CISA_10450442_01|M_Webshell_LEMURLOOT_.*|MOVEit_Transfer_exploit_.*|win_xworm_w0|Windows_Trojan_Xworm_732e6c12"}[5m])',
                            "legendFormat": "{{rule}}"
                        }
                    ],
                    "yaxes": [{"format": "short"}],
                    "alert": {
                        "alertRuleTags": {"team": "VelmaAgency"},
                        "conditions": [
                            {
                                "evaluator": {"params": [1], "type": "gt"},
                                "operator": {"type": "and"},
                                "query": {"params": ["A", "5m", "now"]},
                                "reducer": {"type": "sum"},
                                "type": "query"
                            }
                        ],
                        "executionErrorState": "alerting",
                        "for": "5m",
                        "frequency": "1m",
                        "handler": 1,
                        "name": "YARA Matches Alert",
                        "notifications": [{"uid": "email_slack"}]  # Configure in Grafana
                    }
                },
                {
                    "type": "histogram",
                    "title": "YARA Scan Latency Distribution",
                    "targets": [
                        {
                            "expr": 'aegisnet_yara_scan_latency_seconds_bucket',
                            "legendFormat": "Latency Bucket"
                        }
                    ],
                    "yaxes": [{"format": "s"}]
                },
                {
                    "type": "heatmap",
                    "title": "CPU and Memory Usage",
                    "targets": [
                        {"expr": 'aegisnet_cpu_usage_percent', "legendFormat": "CPU"},
                        {"expr": 'aegisnet_memory_usage_mb', "legendFormat": "Memory"}
                    ],
                    "yAxis": {"format": "percent"}
                },
                {
                    "type": "table",
                    "title": "Threat Details",
                    "targets": [
                        {
                            "expr": 'aegisnet_yara_matches_total',
                            "format": "table",
                            "legendFormat": "{{rule}}"
                        }
                    ],
                    "transformations": [{"id": "organize", "options": {"indexByName": {"rule": 0, "value": 1}}]
                },
                {
                    "type": "gauge",
                    "title": "Bio-Triad Recovery Rate",
                    "targets": [
                        {"expr": 'aegisnet_recovery_success', "legendFormat": "Recovery"}
                    ],
                    "valueMappings": [{"value": 1, "text": "Success"}],
                    "alert": {
                        "conditions": [
                            {
                                "evaluator": {"params": [0], "type": "lt"},
                                "operator": {"type": "and"},
                                "query": {"params": ["A", "5m", "now"]},
                                "reducer": {"type": "last"},
                                "type": "query"
                            }
                        ],
                        "name": "Recovery Failure Alert",
                        "notifications": [{"uid": "email_slack"}]
                    }
                },
                {
                    "type": "stat",
                    "title": "Anomaly Score",
                    "targets": [
                        {"expr": 'aegisnet_anomaly_score', "legendFormat": "Score"}
                    ],
                    "thresholds": [{"value": 0.93, "color": "red"}],
                    "alert": {
                        "conditions": [
                            {
                                "evaluator": {"params": [0.93], "type": "gt"},
                                "operator": {"type": "and"},
                                "query": {"params": ["A", "5m", "now"]},
                                "reducer": {"type": "last"},
                                "type": "query"
                            }
                        ],
                        "name": "High Anomaly Score Alert",
                        "notifications": [{"uid": "email_slack"}]
                    }
                }
            ]
        },
        "overwrite": True
    }
    return dashboard

@app.route('/metrics')
def metrics():
    """Expose Prometheus metrics for Grafana."""
    try:
        shield = APTShield()
        triad = BioTriad()
        data = b"MOVEit.DMZ X-siLock-Comment"
        input_tensor = torch.rand(128)
        
        matches = shield.scan_data(data)
        for match in matches:
            yara_matches.labels(rule=match).inc()
        recovery_success.set(1 if triad.planarian_healing(input_tensor, 0.95, 1.0) else 0)
        anomaly_score.set(0.95)
        cpu_usage.set(1.0)  # Mock
        memory_usage.set(90.0)  # Mock
        return Response(generate_latest(), mimetype='text/plain')
    except Exception as e:
        logger.error(f"Metrics error: {e}")
        return Response("Error", status=500)

@app.route('/dashboard')
def get_dashboard():
    """Return Grafana dashboard JSON."""
    try:
        dashboard = setup_grafana_dashboard()
        return Response(json.dumps(dashboard), mimetype='application/json')
    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        return Response("Error", status=500)

if __name__ == "__main__":
    from prometheus_client import start_http_server
    start_http_server(8000)
    app.run(host='0.0.0.0', port=5000)
    # frontend.py - Grafana Dashboard with Alertmanager Integration for AegisNet v2.1.1
from flask import Flask, Response
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from core import APTShield, BioTriad
import torch
import json
import logging
import requests
import os
from typing import Dict

logger = logging.getLogger(__name__)
app = Flask(__name__)

# Prometheus metrics
yara_matches = Counter('aegisnet_yara_matches_total', 'Total YARA rule matches', ['rule'])
yara_latency = Histogram('aegisnet_yara_scan_latency_seconds', 'YARA scan latency')
recovery_success = Gauge('aegisnet_recovery_success', 'Bio-Triad recovery success rate')
anomaly_score = Gauge('aegisnet_anomaly_score', 'Anomaly detection score')
cpu_usage = Gauge('aegisnet_cpu_usage_percent', 'CPU usage')
memory_usage = Gauge('aegisnet_memory_usage_mb', 'Memory usage')

def setup_alertmanager_config() -> Dict:
    """Generate Alertmanager configuration using environment variables."""
    config = {
        "global": {
            "smtp_smarthost": os.getenv("SMTP_HOST", "smtp.example.com:587"),
            "smtp_from": os.getenv("SMTP_FROM", "alerts@velmaagency.com")
        },
        "route": {
            "receiver": "team-email-slack",
            "group_by": ["alertname", "rule"],
            "group_wait": "30s",
            "group_interval": "5m",
            "repeat_interval": "1h"
        },
        "receivers": [
            {
                "name": "team-email-slack",
                "email_configs": [{"to": os.getenv("ALERT_EMAIL", "team@velmaagency.com")}],
                "slack_configs": [{"api_url": os.getenv("SLACK_API_URL", "")}]
            }
        ]
    }
    return config

def push_to_alertmanager(alert_data: Dict) -> bool:
    """Push alert to Alertmanager."""
    try:
        alertmanager_url = os.getenv("ALERTMANAGER_URL", "http://localhost:9093/api/v2/alerts")
        headers = {"Content-Type": "application/json"}
        response = requests.post(alertmanager_url, json=[alert_data], headers=headers, timeout=5)
        response.raise_for_status()
        logger.info("Alert pushed to Alertmanager", extra={"alert": alert_data})
        return True
    except Exception as e:
        logger.error(f"Alertmanager push error: {e}")
        return False

def setup_grafana_dashboard() -> Dict:
    """Generate detailed Grafana dashboard JSON with alerting."""
    dashboard = {
        "dashboard": {
            "title": "AegisNet v2.1.1 Cybersecurity Dashboard",
            "panels": [
                {
                    "type": "timeseries",
                    "title": "YARA Rule Matches by Threat",
                    "targets": [
                        {
                            "expr": 'rate(aegisnet_yara_matches_total{rule=~"CISA_10450442_01|M_Webshell_LEMURLOOT_.*|MOVEit_Transfer_exploit_.*|win_xworm_w0|Windows_Trojan_Xworm_732e6c12"}[5m])',
                            "legendFormat": "{{rule}}"
                        }
                    ],
                    "yaxes": [{"format": "short"}],
                    "alert": {
                        "alertRuleTags": {"team": "VelmaAgency"},
                        "conditions": [
                            {
                                "evaluator": {"params": [1], "type": "gt"},
                                "operator": {"type": "and"},
                                "query": {"params": ["A", "5m", "now"]},
                                "reducer": {"type": "sum"},
                                "type": "query"
                            }
                        ],
                        "executionErrorState": "alerting",
                        "for": "5m",
                        "frequency": "1m",
                        "handler": 1,
                        "name": "YARA Matches Alert",
                        "message": "High YARA matches detected: {{ $labels.rule }}",
                        "notifications": [{"uid": "alertmanager"}]
                    }
                },
                {
                    "type": "histogram",
                    "title": "YARA Scan Latency Distribution",
                    "targets": [
                        {
                            "expr": 'aegisnet_yara_scan_latency_seconds_bucket',
                            "legendFormat": "Latency Bucket"
                        }
                    ],
                    "yaxes": [{"format": "s"}]
                },
                {
                    "type": "heatmap",
                    "title": "CPU and Memory Usage",
                    "targets": [
                        {"expr": 'aegisnet_cpu_usage_percent', "legendFormat": "CPU"},
                        {"expr": 'aegisnet_memory_usage_mb', "legendFormat": "Memory"}
                    ],
                    "yAxis": {"format": "percent"}
                },
                {
                    "type": "table",
                    "title": "Threat Details",
                    "targets": [
                        {
                            "expr": 'aegisnet_yara_matches_total',
                            "format": "table",
                            "legendFormat": "{{rule}}"
                        }
                    ],
                    "transformations": [{"id": "organize", "options": {"indexByName": {"rule": 0, "value": 1}}}]
                },
                {
                    "type": "gauge",
                    "title": "Bio-Triad Recovery Rate",
                    "targets": [
                        {"expr": 'aegisnet_recovery_success', "legendFormat": "Recovery"}
                    ],
                    "valueMappings": [{"value": 1, "text": "Success"}],
                    "alert": {
                        "conditions": [
                            {
                                "evaluator": {"params": [0], "type": "lt"},
                                "operator": {"type": "and"},
                                "query": {"params": ["A", "5m", "now"]},
                                "reducer": {"type": "last"},
                                "type": "query"
                            }
                        ],
                        "name": "Recovery Failure Alert",
                        "message": "Bio-Triad recovery failed",
                        "notifications": [{"uid": "alertmanager"}]
                    }
                },
                {
                    "type": "stat",
                    "title": "Anomaly Score",
                    "targets": [
                        {"expr": 'aegisnet_anomaly_score', "legendFormat": "Score"}
                    ],
                    "thresholds": [{"value": 0.93, "color": "red"}],
                    "alert": {
                        "conditions": [
                            {
                                "evaluator": {"params": [0.93], "type": "gt"},
                                "operator": {"type": "and"},
                                "query": {"params": ["A", "5m", "now"]},
                                "reducer": {"type": "last"},
                                "type": "query"
                            }
                        ],
                        "name": "High Anomaly Score Alert",
                        "message": "High anomaly score: {{ $value }}",
                        "notifications": [{"uid": "alertmanager"}]
                    }
                }
            ]
        },
        "overwrite": True
    }
    return dashboard

@app.route('/metrics')
def metrics():
    """Expose Prometheus metrics for Grafana and Alertmanager."""
    try:
        shield = APTShield()
        triad = BioTriad()
        data = b"MOVEit.DMZ X-siLock-Comment"
        input_tensor = torch.rand(128)
        
        matches = shield.scan_data(data)
        for match in matches:
            yara_matches.labels(rule=match).inc()
            alert_data = {
                "labels": {"alertname": "YARA Matches", "rule": match, "severity": "critical"},
                "annotations": {"summary": f"YARA match detected: {match}"}
            }
            push_to_alertmanager(alert_data)
        
        recovery_result = triad.planarian_healing(input_tensor, 0.95, 1.0)
        recovery_success.set(1 if recovery_result else 0)
        if not recovery_result:
            alert_data = {
                "labels": {"alertname": "Recovery Failure", "severity": "warning"},
                "annotations": {"summary": "Bio-Triad recovery failed"}
            }
            push_to_alertmanager(alert_data)
        
        anomaly_score.set(0.95)
        if 0.95 > 0.93:
            alert_data = {
                "labels": {"alertname": "High Anomaly Score", "severity": "critical"},
                "annotations": {"summary": f"High anomaly score: 0.95"}
            }
            push_to_alertmanager(alert_data)
        
        cpu_usage.set(1.0)  # Mock
        memory_usage.set(90.0)  # Mock
        return Response(generate_latest(), mimetype='text/plain')
    except Exception as e:
        logger.error(f"Metrics error: {e}")
        return Response("Error", status=500)

@app.route('/dashboard')
def get_dashboard():
    """Return Grafana dashboard JSON."""
    try:
        dashboard = setup_grafana_dashboard()
        return Response(json.dumps(dashboard), mimetype='application/json')
    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        return Response("Error", status=500)

@app.route('/alertmanager-config')
def get_alertmanager_config():
    """Return Alertmanager configuration JSON."""
    try:
        config = setup_alertmanager_config()
        return Response(json.dumps(config), mimetype='application/json')
    except Exception as e:
        logger.error(f"Alertmanager config error: {e}")
        return Response("Error", status=500)

if __name__ == "__main__":
    from prometheus_client import start_http_server
    start_http_server(8000)
    app.run(host='0.0.0.0', port=5000)
    
    