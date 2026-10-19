"""Latency benchmark on an otherwise idle machine -> results/latency.json.

    python run_latency.py --model models/saved/rt_mlids.pkl

Measures (a) model-only batch inference (batch of 512) and (b) a Kafka-free
end-to-end replay through the streaming pipeline (preprocessing + inference +
alerting), both in microseconds per flow.
"""
import argparse, json, os, platform, time, warnings
from types import SimpleNamespace

import joblib
import numpy as np

from src.data import load_nsl_kdd
from src.metrics import latency_us_per_flow
from src.pipeline.stream_processor import RTMLIDSPipeline

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="models/saved/rt_mlids.pkl")
ap.add_argument("--data-path", default="data/NSL_KDD")
ap.add_argument("--out", default="results/latency.json")
ap.add_argument("--n-flows", type=int, default=5000)
a = ap.parse_args()

b = joblib.load(a.model); ens, prep = b["model"], b["preprocessor"]
_, test = load_nsl_kdd(a.data_path)
X = prep.transform(test)
for m in (ens.rf, ens.xgb, ens):
    m.n_jobs = 1 if hasattr(m, "n_jobs") else None
out = {"machine": {"cpus": os.cpu_count(), "platform": platform.platform()}, "model_only_us_per_flow": {
    "Random Forest": latency_us_per_flow(ens.rf.predict, X),
    "XGBoost": latency_us_per_flow(ens.xgb.predict, X),
    "Stacked Ensemble": latency_us_per_flow(ens.predict, X)}}

rows = test.drop(columns=["label"]).head(a.n_flows).to_dict("records")
msgs = [SimpleNamespace(value={"features": r}) for r in rows]
pipe = RTMLIDSPipeline(ens, prep, consumer=iter(msgs), buffer_size=512)
t0 = time.perf_counter(); pipe.run(); dt = time.perf_counter() - t0
out["end_to_end_replay"] = {"flows": len(rows), "us_per_flow": dt / len(rows) * 1e6,
                            "alerts": pipe.alert_engine.alert_count,
                            "note": "preprocessing + inference + alerting; no Kafka transport, no feature extraction"}
json.dump(out, open(a.out, "w"), indent=2); print(json.dumps(out, indent=2))
