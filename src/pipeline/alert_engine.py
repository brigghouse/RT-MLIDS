"""Alert generation: structured log records for confident non-normal predictions."""

import logging
from datetime import datetime, timezone

logger = logging.getLogger("rt-mlids.alerts")

SEVERITY = {"u2r": "CRITICAL", "r2l": "HIGH", "dos": "HIGH", "probe": "MEDIUM"}


class AlertEngine:
    def __init__(self, threshold: float = 0.85):
        self.threshold = threshold
        self.alert_count = 0
        self.alerts: list[dict] = []

    def fire(self, label: str, confidence: float, flow_idx: int) -> dict:
        self.alert_count += 1
        alert = {
            "id": self.alert_count,
            "severity": SEVERITY.get(label.lower(), "INFO"),
            "category": label,
            "confidence": float(confidence),
            "flow": flow_idx,
            "time": datetime.now(timezone.utc).isoformat(),
        }
        self.alerts.append(alert)
        logger.warning("[ALERT #%d] %s | Category: %s | Confidence: %.4f | Flow: %d | Time: %s",
                       alert["id"], alert["severity"], label, confidence, flow_idx, alert["time"])
        return alert
