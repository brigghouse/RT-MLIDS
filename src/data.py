"""NSL-KDD loading, label mapping and preprocessing.

Pipeline (fitted on the training split only):
    41 raw features -> one-hot (protocol_type, service, flag) -> 122 columns
    -> Mutual-Information top-k (default 30) -> MinMax scaling to [0, 1]
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import MinMaxScaler

KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty",
]
CATEGORICAL = ["protocol_type", "service", "flag"]

# Class index == position here == alphabetical order (what LabelEncoder gives).
CLASS_NAMES = ["dos", "normal", "probe", "r2l", "u2r"]
CLASS_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}
NORMAL_INDEX = CLASS_INDEX["normal"]

ATTACK_MAP = {
    "normal": "normal",
    # DoS
    "back": "dos", "land": "dos", "neptune": "dos", "pod": "dos",
    "smurf": "dos", "teardrop": "dos", "apache2": "dos", "udpstorm": "dos",
    "processtable": "dos", "worm": "dos", "mailbomb": "dos",
    # Probe
    "ipsweep": "probe", "nmap": "probe", "portsweep": "probe",
    "satan": "probe", "mscan": "probe", "saint": "probe",
    # R2L
    "ftp_write": "r2l", "guess_passwd": "r2l", "imap": "r2l",
    "multihop": "r2l", "phf": "r2l", "spy": "r2l", "warezclient": "r2l",
    "warezmaster": "r2l", "sendmail": "r2l", "named": "r2l",
    "snmpgetattack": "r2l", "snmpguess": "r2l", "xlock": "r2l",
    "xsnoop": "r2l", "httptunnel": "r2l",
    # U2R
    "buffer_overflow": "u2r", "loadmodule": "u2r", "perl": "u2r",
    "rootkit": "u2r", "ps": "u2r", "sqlattack": "u2r", "xterm": "u2r",
}


def map_labels(raw_labels: pd.Series) -> np.ndarray:
    """Map raw attack names to class indices; raise on any unknown name."""
    names = raw_labels.astype(str).str.rstrip(".")
    unknown = sorted(set(names) - set(ATTACK_MAP))
    if unknown:
        raise ValueError(f"Unmapped NSL-KDD attack labels: {unknown}")
    return names.map(ATTACK_MAP).map(CLASS_INDEX).to_numpy(dtype=int)


def load_nsl_kdd(data_path: str):
    """Return (train_df, test_df) with the difficulty column removed."""
    from pathlib import Path

    p = Path(data_path)
    train = pd.read_csv(p / "KDDTrain+.txt", names=KDD_COLUMNS)
    test = pd.read_csv(p / "KDDTest+.txt", names=KDD_COLUMNS)
    return train.drop(columns=["difficulty"]), test.drop(columns=["difficulty"])


class NSLKDDPreprocessor:
    """One-hot -> MIG top-k -> MinMax. Fit on training data only.

    The category vocabulary (needed so train and test share the same 122
    one-hot columns) may be taken from several frames; this uses no labels.
    """

    def __init__(self, k: int = 30, random_state: int = 42):
        self.k = k
        self.random_state = random_state

    def _onehot(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df.drop(columns=[c for c in ("label", "difficulty") if c in df])
        X = pd.get_dummies(X, columns=CATEGORICAL, dtype=float)
        return X.reindex(columns=self.all_columns_, fill_value=0.0)

    def fit(self, train_df: pd.DataFrame, y: np.ndarray, vocab_frames=()):
        frames = [train_df, *vocab_frames]
        cols = pd.get_dummies(
            pd.concat([f.drop(columns=[c for c in ("label", "difficulty") if c in f])
                       for f in frames]),
            columns=CATEGORICAL, dtype=float,
        ).columns
        self.all_columns_ = list(cols)
        X = self._onehot(train_df)
        self.mi_scores_ = mutual_info_classif(X, y, random_state=self.random_state)
        order = np.argsort(self.mi_scores_)[::-1][: self.k]
        self.selected_idx_ = order
        self.selected_features_ = [self.all_columns_[i] for i in order]
        self.scaler_ = MinMaxScaler().fit(X.iloc[:, order])
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        X = self._onehot(df).iloc[:, self.selected_idx_]
        return self.scaler_.transform(X).astype(np.float32)

    @property
    def categorical_selected_(self) -> list[int]:
        """Indices (within the selected features) that are one-hot columns."""
        pref = tuple(f"{c}_" for c in CATEGORICAL)
        return [i for i, n in enumerate(self.selected_features_) if n.startswith(pref)]
