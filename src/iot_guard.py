import logging
import asyncio
import time
import json
import numpy as np
from typing import Dict, Any
import torch
import cv2
from PIL import Image
from stegano import lsb
from collections import Counter
from math import log
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from mdm_guard import MdmGuard
from scap_engine import ScapEngine
from nova_engine import NovaEngine
from modules.bio_triad.ant_mesh import AntMesh
from modules.bio_triad.planarian_healing import PlanarianHealing
from modules.defenses.tv_guard import TVGuard

logging.basicConfig(filename='aegisnet_iot.log', level=logging.INFO, format='%(asctime)s %(message)s')

class IoTGuard:
    def __init__(self, config: Dict):
        self.config = config
        self.threat_threshold = config.get('threat_threshold', 0.94)
        self.stego_threshold = config.get('stego_threshold', 7.5)
        self.model = SentenceTransformer(config.get('ssr_embedding_model', 'sentence-transformers/all-MiniLM-L6-v2'))
        self.mdm_guard = MdmGuard(config)
        self.scap_engine = ScapEngine(config.get('oval_path', 'scap_content/linux-iot-oval.xml'),
                                      config.get('xccdf_path', 'scap_content/xccdf-iot-benchmark.xml'))
        self.nova_engine = NovaEngine(config, config.get('nova_auth', {}))
        self.tv_guard = TVGuard(config)
        self.ant_mesh = AntMesh()  # Placeholder
        self.planarian = PlanarianHealing()  # Placeholder
        logger.info("IoTGuard initialized with MDM/UEM, SCAP/Nova, steganography, and bio-triad.")

    async def detect_iot_threat(self, signal_data: Dict) -> Dict:
        try:
            start_time = time.perf_counter()
            anomaly_score = 0.0

            # MDM/UEM policy check
            policy_result = await self.mdm_guard.enforce_policy(signal_data['device_id'], 'mfa_enable')
            if policy_result['status'] != 'success':
                return {"status": "non_compliant", "details": policy_result}

            # PCR integrity check
            for i in range(24):
                if not await self.mdm_guard.check_pcr_integrity(i):
                    return {"status": "integrity_fail", "index": i}

            # LUKS/Argon2 KDF
            salt = b'random_salt'  # Replace with secure generation
            key = await self.mdm_guard.tune_kdf(signal_data.get('password', ''), salt, is_edge=True)
            if not key:
                return {"status": "kdf_fail"}

            # SCAP/OVAL compliance
            scan_result = await self.scap_engine.run_scan(signal_data['device_id'], signal_data)
            if scan_result['status'] != 'compliant':
                return scan_result

            # faulTPM mitigation
            if not await self.mdm_guard.mitigate_faul_tpm(signal_data.get('voltage_data', {'spike': 0.0})):
                return {"status": "faul_tpm_detected"}

            # IoPC detection
            iopc_result = await self.nova_engine.detect_iopc(signal_data.get('prompt', ''))
            if iopc_result['status'] == 'anomaly':
                anomaly_score = max(anomaly_score, iopc_result['details'].get('score', 0.95))
                await self.nova_engine.provision_isolation_vm(f"iopc_{signal_data['device_id']}")

            # Steganography detection
            if 'artifact_path' in signal_data:
                stego_result = await self.detect_steganography(signal_data['artifact_path'])
                if stego_result['status'] == 'stego_detected':
                    anomaly_score = max(anomaly_score, stego_result['entropy'])

            # TV Guard for DVB-T2
            if signal_data.get('type') == 'dvb-t2' and self.tv_guard.detect_spoofing(signal_data['signal']):
                anomaly_score = max(anomaly_score, 0.95)

            result = {
                'type': signal_data.get('type', 'unknown'),
                'anomaly_score': anomaly_score,
                'status': 'clean',
                'latency': time.perf_counter() - start_time,
                'timestamp': '2025-10-24T10:28:00Z'
            }
            if anomaly_score > self.threat_threshold:
                self.ant_mesh.isolate_node(signal_data['device_id'])
                self.planarian.repair({'node': signal_data['device_id'], 'pathways': ['PI3K', 'RA']})
                result['status'] = 'isolated_and_repaired'

            logger.info(f"IoT threat scan: {json.dumps(result)}")
            return result
        except Exception as e:
            logger.error(f"IoT threat detection failed: {str(e)}")
            return {"status": "error", "message": str(e), "timestamp": "2025-10-24T10:28:00Z"}

    async def detect_steganography(self, artifact_path: str) -> Dict:
        try:
            start_time = time.perf_counter()
            if artifact_path.endswith(('.mp4', '.avi', '.h264')):
                cap = cv2.VideoCapture(artifact_path)
                max_entropy = 0
                for _ in range(10):  # Sample 10 frames
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    hidden_data = lsb.reveal(frame_pil)
                    if hidden_data:
                        entropy = self.compute_entropy(hidden_data)
                        max_entropy = max(max_entropy, entropy)
                cap.release()
                status = 'stego_detected' if max_entropy > self.stego_threshold else 'clean'
                return {"status": status, "entropy": max_entropy}
            elif artifact_path.endswith(('.jpg', '.png')):
                img = Image.open(artifact_path)
                hidden_data = lsb.reveal(img)
                entropy = self.compute_entropy(hidden_data) if hidden_data else 0
                status = 'stego_detected' if entropy > self.stego_threshold else 'clean'
                return {"status": status, "entropy": entropy}
            elif artifact_path.endswith('.txt'):
                with open(artifact_path, 'r') as f:
                    text = f.read()
                entropy = self.compute_entropy(text)
                status = 'stego_detected' if entropy > self.stego_threshold else 'clean'
                return {"status": status, "entropy": entropy}
            return {"status": "error", "message": "Unsupported artifact type"}
        except Exception as e:
            logger.error(f"Steganography detection failed: {str(e)}")
            return {"status": "error", "message": str(e)}
        finally:
            logger.info(f"Steganography latency: {(time.perf_counter() - start_time):.6f}s")

    def compute_entropy(self, data: str) -> float:
        if not data:
            return 0.0
        counts = Counter(data)
        total = len(data)
        entropy = -sum((count / total) * log(count / total, 2) for count in counts.values())
        return entropy
