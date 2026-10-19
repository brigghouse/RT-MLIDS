import numpy as np
import pandas as pd
import pytest

from src.data import ATTACK_MAP, CLASS_NAMES, NSLKDDPreprocessor, map_labels


def test_class_order_is_alphabetical():
    assert CLASS_NAMES == sorted(CLASS_NAMES)


def test_all_test_set_attacks_are_mapped():
    # attack names that only appear in KDDTest+ (previously fell through to "u2r")
    assert ATTACK_MAP["mailbomb"] == "dos"
    assert ATTACK_MAP["snmpgetattack"] == "r2l"
    assert ATTACK_MAP["saint"] == "probe"


def test_unknown_label_raises():
    with pytest.raises(ValueError):
        map_labels(pd.Series(["normal", "definitely_new_attack"]))


def test_map_labels_indices():
    y = map_labels(pd.Series(["normal", "neptune", "satan", "guess_passwd", "rootkit"]))
    assert [CLASS_NAMES[i] for i in y] == ["normal", "dos", "probe", "r2l", "u2r"]


def test_preprocessor_shape_range_and_no_leak(synthetic_nsl):
    df = synthetic_nsl
    y = map_labels(df["label"])
    prep = NSLKDDPreprocessor(k=10).fit(df.iloc[:150], y[:150], vocab_frames=[df.iloc[150:]])
    X_tr, X_te = prep.transform(df.iloc[:150]), prep.transform(df.iloc[150:])
    assert X_tr.shape[1] == X_te.shape[1] == 10
    assert X_tr.min() >= 0 and X_tr.max() <= 1 + 1e-6
    assert len(prep.selected_features_) == 10
    assert isinstance(prep.categorical_selected_, list)
