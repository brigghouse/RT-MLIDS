import numpy as np

from src.adversarial.attacks import project_to_valid


def test_projection_makes_records_valid():
    adv = np.array([[1.4, -0.2, 0.7, 0.3]])
    out = project_to_valid(adv, [2, 3])
    assert out.min() >= 0 and out.max() <= 1
    assert set(out[0, [2, 3]]) <= {0.0, 1.0}
    assert out[0, 0] == 1.0 and out[0, 1] == 0.0
