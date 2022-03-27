from abc import ABC, abstractmethod
from typing import Any, Collection, Dict

import evaluation


class ThresholdScheduler(ABC):
    def grid_search(
        self,
        attr_values: Collection,
        run: evaluation.Run,
        metric: evaluation.Metric,
        minimize: bool,
    ) -> Dict[str, Any]:
        if len(attr_values) == 0:
            return {}
        for attr in attr_values:
            assert hasattr(
                self, attr
            ), f"{self.__class__.__name__} has no attribute {attr}"
        attr, values = attr_values.popitem()

        sign = -1 if minimize else 1
        best_value = getattr(self, attr)
        best_result = sign * metric(run)

        for value in values:
            setattr(self, attr, value)
            for predictions in run.values():
                for i, (_, score) in enumerate(predictions):
                    _, score = predictions[i]
                    predictions[i] = (self.decide(score, i), score)
            result = sign * metric(run)
            if result > best_result:
                best_value = self.threshold
                best_result = result

        setattr(self, attr, best_value)
        best_attr_values = self.grid_search(attr_values, run, metric, minimize)
        best_attr_values[attr] = best_value
        return best_attr_values

    @abstractmethod
    def decide(self, score: float, round: int) -> bool:
        raise NotImplementedError()


class ConstantThresholdScheduler(ThresholdScheduler):
    def __init__(self, threshold: float):
        self.threshold = threshold

    def decide(self, score: float, round: int) -> bool:
        return score >= self.threshold


class ExponentialThresholdScheduler(ThresholdScheduler):
    def __init__(
        self, start_threshold: float, target_threshold: float, time_constant: float
    ):
        """
        Exponentially approach a target threshold.

        Args:
            start_threshold: Threshold for the first post.
            target_threshold: Threshold for late posts (asymptotically approached).
            time_constant: Number of posts after which the threshold has almost
                reached `target_threshold` (90% of the way).
        """
        self.start_threshold = start_threshold
        self.target_threshold = target_threshold
        self.time_constant = time_constant

    def decide(self, score: float, round: int) -> bool:
        threshold = self.target_threshold + (
            self.start_threshold - self.target_threshold
        ) * 10 ** (-round / self.time_constant)
        return bool(score >= threshold)
