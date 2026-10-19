"""Streaming pipeline: record source (Kafka or any iterable) -> preprocessing ->
ensemble inference -> alerts.

Records are JSON objects. Either ``{"features": {<raw NSL-KDD column>: value, ...}}``
(preprocessed here with the fitted ``NSLKDDPreprocessor``) is expected.
``kafka-python`` is imported lazily, so the pipeline can be tested and replayed
without a broker by passing any iterable of message-like objects as ``consumer``.
"""

from __future__ import annotations

import json
import logging
import time

import numpy as np
import pandas as pd

from src.data import CLASS_NAMES, NORMAL_INDEX
from src.pipeline.alert_engine import AlertEngine

logger = logging.getLogger("rt-mlids")


class RTMLIDSPipeline:
    def __init__(self, model, preprocessor, consumer=None, kafka_broker=None, topic=None,
                 confidence_threshold: float = 0.85, buffer_size: int = 512):
        self.model = model
        self.preprocessor = preprocessor
        self.buffer_size = buffer_size
        self.alert_engine = AlertEngine(threshold=confidence_threshold)
        self.threshold = confidence_threshold
        if consumer is None:
            from kafka import KafkaConsumer  # lazy: optional dependency

            consumer = KafkaConsumer(
                topic, bootstrap_servers=[kafka_broker],
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                auto_offset_reset="latest", enable_auto_commit=True)
        self.consumer = consumer
        self._buffer: list[dict] = []
        self.batch_latencies_ms: list[float] = []

    def run(self):
        """Consume until the source is exhausted (a Kafka consumer never is)."""
        for message in self.consumer:
            self._buffer.append(message.value["features"])
            if len(self._buffer) >= self.buffer_size:
                self.flush()
        self.flush()

    def flush(self):
        if not self._buffer:
            return []
        rows, self._buffer = self._buffer, []
        t0 = time.perf_counter()
        X = self.preprocessor.transform(pd.DataFrame(rows))
        alerts = self._process_batch(X)
        self.batch_latencies_ms.append((time.perf_counter() - t0) * 1e3)
        return alerts

    def _process_batch(self, X: np.ndarray):
        preds, conf, ok = self.model.predict_with_confidence(X)
        alerts = []
        for i, (p, c, confident) in enumerate(zip(preds, conf, ok)):
            if confident and p != NORMAL_INDEX:
                alerts.append(self.alert_engine.fire(CLASS_NAMES[int(p)], c, i))
        return alerts
