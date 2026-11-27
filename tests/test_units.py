import numpy as np
from sklearn.metrics import roc_auc_score

from actground.data import extract_letter, extract_yesno, label_answer
from actground.probes import jaccard, linear_cka, neuron_aurocs
from actground.stats import holm, mcnemar_p


def test_extract_letter():
    assert extract_letter("D) Nitrofurantoin") == "D"
    assert extract_letter("The answer is b.") == "B"
    assert extract_letter("  (C) because") == "C"
    assert extract_letter("Nitrofurantoin") == "unknown"
    assert extract_letter("") == "unknown"
    assert extract_letter("Amoxicillin") == "A", "a leading A-D is read as the answer even in free text"


def test_extract_yesno():
    assert extract_yesno("Yes, the study confirms") == "yes"
    assert extract_yesno("The answer is no") == "no"
    assert extract_yesno("It is maybe the case") == "maybe"
    assert extract_yesno("Unclear from the abstract") == "unknown"


def test_label_answer():
    assert label_answer("B", "B", "mcq") == ("B", False)
    assert label_answer("B", "C", "mcq") == ("B", True)
    assert label_answer("no idea", "C", "mcq") == ("unknown", None)


def test_neuron_aurocs_match_sklearn():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 5))
    y = (rng.random(200) < 0.3).astype(int)
    X[:, 0] += y
    X[:, 4] = 1.0
    expected = [max(a, 1 - a) for a in (roc_auc_score(y, X[:, j]) for j in range(4))] + [0.5]
    assert np.allclose(neuron_aurocs(X, y), expected)


def test_cka_and_jaccard():
    X = np.random.default_rng(0).normal(size=(50, 8))
    assert linear_cka(X, 3 * X) > 0.999
    assert jaccard([1, 2, 3], [2, 3, 4]) == 0.5


def test_mcnemar_and_holm():
    assert mcnemar_p(0, 0) == 1.0
    assert mcnemar_p(10, 0) < 0.01
    assert mcnemar_p(30, 30) > 0.85
    assert holm([0.01, 0.04, 0.03, None])[:3] == [0.03, 0.06, 0.06]
