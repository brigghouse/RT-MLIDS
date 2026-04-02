"""
RT-MLIDS Real-Time Streaming Pipeline
Consumes network flow records from Apache Kafka and runs ensemble inference.
"""

import json
import logging
import numpy as np
from kafka import KafkaConsumer
from src.models.ensemble import StackedEnsemble
from src.pipeline.alert_engine import AlertEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rt-mlids")

LABEL_MAP = {0: "Normal", 1: "DoS", 2: "Probe", 3: "R2L", 4: "U2R"}


class RTMLIDSPipeline:
    """
    End-to-end streaming pipeline:
    Kafka → Feature buffer → Ensemble inference → Alert generation
    """

    def __init__(
        self,
        kafka_broker: str,
        topic: str,
        model_path: str = "models/saved/rt_mlids.pkl",
        confidence_threshold: float = 0.85,
        buffer_size: int = 512,
    ):
        self.topic = topic
        self.buffer_size = buffer_size
        self.model = StackedEnsemble.load(model_path)
        self.alert_engine = AlertEngine(threshold=confidence_threshold)
        self.consumer = KafkaConsumer(
            topic,
            bootstrap_servers=[kafka_broker],
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            auto_offset_reset="latest",
            enable_auto_commit=True,
        )
        self._buffer: list[np.ndarray] = []
        logger.info("RT-MLIDS pipeline initialised. Listening on topic: %s", topic)

    def run(self):
        """Main loop: consume, buffer, infer, alert."""
        for message in self.consumer:
            flow = np.array(message.value["features"], dtype=np.float32)
            self._buffer.append(flow)

            if len(self._buffer) >= self.buffer_size:
                X_batch = np.vstack(self._buffer)
                self._buffer.clear()
                self._process_batch(X_batch)

    def _process_batch(self, X: np.ndarray):
        predictions, confidences, confident_mask = (
            self.model.predict_with_confidence(X)
        )
        for i, (pred, conf, is_confident) in enumerate(
            zip(predictions, confidences, confident_mask)
        ):
            label = LABEL_MAP.get(pred, "Unknown")
            if is_confident and label != "Normal":
                self.alert_engine.fire(label=label, confidence=conf, flow_idx=i)
