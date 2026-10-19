"""Stream processor tested with a fake consumer (no Kafka broker needed)."""

from types import SimpleNamespace

import numpy as np

from src.data import NSLKDDPreprocessor, map_labels
from src.models.ensemble import StackedEnsemble
from src.pipeline.stream_processor import RTMLIDSPipeline


def test_pipeline_alerts_on_attacks_with_correct_labels(synthetic_nsl):
    df = synthetic_nsl
    y = map_labels(df["label"])
    prep = NSLKDDPreprocessor(k=10).fit(df, y)
    model = StackedEnsemble(n_estimators_rf=30, max_depth_rf=6, n_estimators_xgb=30, n_folds=3,
                            n_jobs=1, confidence_threshold=0.5).fit(prep.transform(df), y)
    msgs = [SimpleNamespace(value={"features": row.drop("label").to_dict()}) for _, row in df.head(40).iterrows()]
    pipe = RTMLIDSPipeline(model, prep, consumer=iter(msgs), confidence_threshold=0.5, buffer_size=16)
    pipe.run()
    assert len(pipe.batch_latencies_ms) == 3  # 16 + 16 + 8
    cats = {a["category"] for a in pipe.alert_engine.alerts}
    assert cats and cats.issubset({"dos", "probe", "r2l", "u2r"})  # never alerts on "normal"
    # alerts for known-DoS rows should be labelled dos, not mislabelled by an index shift
    dos_rows = int((df.head(40)["label"] == "neptune").sum())
    assert sum(a["category"] == "dos" for a in pipe.alert_engine.alerts) >= dos_rows // 2
