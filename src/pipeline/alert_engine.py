"""
Alert generation engine for RT-MLIDS.
Fires structured alerts when the ensemble confidence exceeds the threshold.
"""

import logging
from datetime import datetime, timezone

logger = logging.getLogger("rt-mlids.alerts")

SEVERITY = {
    "U2R":   "CRITICAL",
    "R2L":   "HIGH",
    "DoS":   "HIGH",
    "Probe": "MEDIUM",
}


class AlertEngine:
    def __init__(self, threshold: float = 0.85):
        self.threshold = threshold
        self.alert_count = 0

    def fire(self, label: str, confidence: float, flow_idx: int):
        self.alert_count += 1
        severity = SEVERITY.get(label, "INFO")
        ts = datetime.now(timezone.utc).isoformat()
        logger.warning(
            "[ALERT #%d] %s | Category: %s | Confidence: %.4f | Flow: %d | Time: %s",
            self.alert_count, severity, label, confidence, flow_idx, ts,
        )
