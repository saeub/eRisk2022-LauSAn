from typing import Collection, Dict, Sequence

import numpy as np

from data import Subject


def erde(decisions: Sequence[bool], truth: bool, o: float = 5) -> float:
    assert isinstance(truth, bool) and all(
        isinstance(d, bool) for d in decisions
    ), "Decision and truth values must be booleans"

    c_fn = 1
    c_fp = 214 / 1707  # proportion of positive subjects
    c_tp = 1  # late detection is equivalent to not detecting at all

    def lc_o(k: int) -> float:
        return 1 - 1 / (1 + np.exp(k - o))

    try:
        decision_time = decisions.index(True)
        decision = True
    except ValueError:
        decision_time = None
        decision = False

    if decision is True and truth is False:
        return c_fp
    if decision is False and truth is True:
        return c_fn
    if decision is True and truth is True:
        return c_tp * lc_o(decision_time)
    if decision is False and truth is False:
        return 0.0


def mean_erde(
    decisions: Dict[str, Sequence[bool]], subjects: Collection[Subject], o: float = 5
) -> float:
    results = []
    for subject in subjects:
        results.append(erde(decisions[subject.id], subject.label, o=o))
    return np.mean(results)


def recall_precision_f1(
    decisions: Dict[str, Sequence[bool]], subjects: Collection[Subject]
) -> float:
    tp = 0
    fp = 0
    fn = 0
    for subject in subjects:
        decision = any(decision is True for decision in decisions[subject.id])
        truth = subject.label

        if decision is True and truth is True:
            tp += 1
        elif decision is True and truth is False:
            fp += 1
        elif decision is False and truth is True:
            fn += 1

    recall = tp / ((tp + fn) or 1)
    precision = tp / ((tp + fp) or 1)
    f1 = 2 * tp / ((2 * tp + fp + fn) or 1)
    return recall, precision, f1
