import logging
import asyncio
import openscap.API as oscap
from typing import Dict

logger = logging.getLogger(__name__)

class ScapEngine:
    def __init__(self, oval_file: str = "scap_content/linux-iot-oval.xml", xccdf_file: str = "scap_content/xccdf-iot-benchmark.xml"):
        self.oval_ctx = oscap.oval_new_context()
        self.scap_source = oscap.source_new_from_file(xccdf_file)
        self.oval_defs = oscap.oval_import_source(oval_file)
        self.session = oscap.scap_session_new(self.scap_source)
        logger.info("SCAP/OVAL engine initialized with IoT/fault/DPA/TEMPEST rules.")

    async def run_scan(self, device_id: str, device_state: Dict) -> Dict:
        try:
            probe_data = oscap.oval_probe_data_new()
            for key, value in device_state.items():
                oscap.oval_probe_data_add(probe_data, key, value)

            oval_results = oscap.oval_evaluate_definitions(self.oval_defs, probe_data)
            vuln_score = sum(result.score for result in oval_results if result.result == oscap.OVAL_RESULT_TRUE) / len(oval_results) if oval_results else 0

            self.session.load()
            scap_results = self.session.evaluate()
            compliance_score = scap_results.get_compliance_percentage()

            if compliance_score < 99.9 or vuln_score > 0.1:
                logger.warning("SCAP/OVAL non-compliance or vulnerability", extra={"device_id": device_id, "compliance": compliance_score, "vuln_score": vuln_score})
                return {"status": "non_compliant", "details": {"compliance": compliance_score, "vulnerabilities": [r.id for r in oval_results if r.result == oscap.OVAL_RESULT_TRUE]}}
            
            logger.info("SCAP/OVAL scan passed", extra={"device_id": device_id})
            return {"status": "compliant", "details": {"compliance": compliance_score, "vuln_score": vuln_score}}
        except Exception as e:
            logger.error("SCAP/OVAL scan error", extra={"error": str(e)})
            return {"status": "error", "details": str(e)}
        finally:
            oscap.oval_probe_data_free(probe_data)
