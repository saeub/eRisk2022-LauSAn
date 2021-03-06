from abc import ABC, abstractmethod
from typing import Any, Collection, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

import evaluation
from log import logger


class ThresholdScheduler(ABC):
    def __init__(self):
        self._grid_search_results = None

    def grid_search(
        self,
        attr_values: Dict[str, Collection[Any]],
        run: evaluation.Run,
        metric: evaluation.Metric,
        minimize: bool,
        fixed_attr_values: Optional[Dict[str, Any]] = None,
        progress: tqdm = None,
    ):
        if len(attr_values) == 0:
            logger.warn(
                f"({self.__class__.__name__}) Model class does not define any"
                "attributes to perform grid search on. Skipping grid search."
            )
            return

        if fixed_attr_values is None:
            # This is the first level of recursion -> reset everything
            self._grid_search_results = []
            fixed_attr_values = {}
            num_configurations = np.prod(
                [len(values) for values in attr_values.values()]
            )
            progress = tqdm(total=num_configurations)

        for attr in attr_values:
            assert hasattr(
                self, attr
            ), f"{self.__class__.__name__} has no attribute {attr}"
        attr_values = {**attr_values}
        attr, values = attr_values.popitem()

        sign = -1 if minimize else 1
        best_value = getattr(self, attr)

        for value in values:
            setattr(self, attr, value)
            current_fixed_attr_values = {**fixed_attr_values, attr: value}
            for predictions in run.values():
                for i, (_, score) in enumerate(predictions):
                    _, score = predictions[i]
                    predictions[i] = (self.decide(score, i), score)
            result = sign * metric(run)

            if len(attr_values) > 0:
                # Fix current value, continue recursion
                self.grid_search(
                    attr_values,
                    run,
                    metric,
                    minimize,
                    current_fixed_attr_values,
                    progress,
                )
            else:
                # This is the last level of recursion -> store result
                self._grid_search_results.append(
                    (current_fixed_attr_values, sign * result)
                )
                progress.update()

        if len(fixed_attr_values) == 0:
            # All configurations tested -> select the configuration with the best result
            best_attr_values, _ = max(
                self._grid_search_results, key=lambda x: sign * x[1]
            )
            for attr, best_value in best_attr_values.items():
                setattr(self, attr, best_value)

    @property
    def grid_search_results(self) -> List[Tuple[Dict[str, Any], float]]:
        return self._grid_search_results

    @abstractmethod
    def decide(self, score: float, round: int) -> bool:
        raise NotImplementedError()


class ConstantThresholdScheduler(ThresholdScheduler):
    def __init__(self, threshold: float):
        super().__init__()
        self.threshold = threshold

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.threshold})"

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
        super().__init__()
        self.start_threshold = start_threshold
        self.target_threshold = target_threshold
        self.time_constant = time_constant

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.start_threshold}, {self.target_threshold}, {self.time_constant})"

    def decide(self, score: float, round: int) -> bool:
        threshold = self.target_threshold + (
            self.start_threshold - self.target_threshold
        ) * 10 ** (-round / self.time_constant)
        return bool(score >= threshold)


class WaitExponentialThresholdScheduler(ExponentialThresholdScheduler):
    def __init__(self, wait: int, *args, **kwargs):
        """
        Like `ExponentialThresholdScheduler`, but with a forced `False` decision in the
        first `wait` rounds. This is to be used with models where we know that the first
        few posts typically result in unreliable decisions.
        """
        super().__init__(*args, **kwargs)
        self.wait = wait

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.wait}, {self.start_threshold}, {self.target_threshold}, {self.time_constant})"

    def decide(self, score: float, round: int) -> bool:
        if round < self.wait:
            return False
        return super().decide(score, round)
