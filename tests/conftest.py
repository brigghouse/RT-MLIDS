import numpy as np
import pandas as pd
import pytest

from src.data import ATTACK_MAP, KDD_COLUMNS


@pytest.fixture
def synthetic_nsl():
    """Small synthetic frame with NSL-KDD's schema (not real traffic)."""
    rng = np.random.default_rng(0)
    names = ["normal"] * 120 + ["neptune"] * 60 + ["satan"] * 30 + ["guess_passwd"] * 20 + ["rootkit"] * 12
    rows = []
    for lab in names:
        r = {c: float(rng.random()) for c in KDD_COLUMNS}
        r.update(protocol_type=rng.choice(["tcp", "udp", "icmp"]),
                 service=rng.choice(["http", "ftp", "private"]),
                 flag=rng.choice(["SF", "S0", "REJ"]), label=lab, difficulty=10)
        # make the class learnable
        r["serror_rate"] = 1.0 if lab == "neptune" else r["serror_rate"] * 0.2
        r["src_bytes"] = 0.0 if lab == "satan" else r["src_bytes"]
        rows.append(r)
    df = pd.DataFrame(rows)[KDD_COLUMNS].drop(columns=["difficulty"])
    return df.sample(frac=1.0, random_state=0).reset_index(drop=True)
