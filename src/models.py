from abc import ABC, abstractmethod
import random
from data import Subject
from typing import Collection, Sequence


class Model(ABC):
    @abstractmethod
    def train(self, subjects: Collection[Subject]):
        raise NotImplementedError()

    @abstractmethod
    def predict(self, subjects: Sequence[Subject]) -> Sequence[float]:
        raise NotImplementedError()


class DummyModel(Model):
    def train(self, subjects: Collection[Subject]):
        pass

    def predict(self, subjects: Sequence[Subject]) -> Sequence[float]:
        return [random.random() for subject in subjects]


def load(filename: str) -> Model:
    # TODO
    return DummyModel()
