from abc import ABC, abstractmethod


class ThresholdScheduler(ABC):
    @abstractmethod
    def decide(self, score: float, round: int) -> bool:
        raise NotImplementedError()


class ConstantThresholdScheduler(ThresholdScheduler):
    def __init__(self, threshold: float):
        self.threshold = threshold

    def decide(self, score: float, round: int) -> bool:
        return score >= 0.5


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
