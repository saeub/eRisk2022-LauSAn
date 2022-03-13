import random
from abc import ABC, abstractmethod
from typing import Collection, Sequence

from data import Subject


class Model(ABC):
    @abstractmethod
    def train(self, subjects: Collection[Subject]):
        raise NotImplementedError()

    @abstractmethod
    def predict(self, subjects: Sequence[Subject]) -> Sequence[float]:
        raise NotImplementedError()


class RandomBaseline(Model):
    def __init__(self, positive_ratio: float = 0.125):
        self.positive_ratio = positive_ratio
        self.subject_predictions = {}

    def train(self, subjects: Collection[Subject]):
        pass

    def predict(self, subjects: Sequence[Subject]) -> Sequence[float]:
        predictions = []
        for subject in subjects:
            predictions.append(
                self.subject_predictions.setdefault(
                    subject.id, float(random.random() < self.positive_ratio)
                )
            )
        return predictions


def load(filename: str) -> Model:
    # TODO
    return RandomBaseline()
