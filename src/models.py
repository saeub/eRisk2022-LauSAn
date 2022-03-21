import pickle
import random
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Collection, List, Sequence, Tuple

import numpy as np
import sklearn.linear_model
import torch
import transformers

from data import Subject
from threshold_schedulers import ThresholdScheduler, ExponentialThresholdScheduler

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


class Model(ABC):
    def __init__(self, threshold_scheduler: ThresholdScheduler = None):
        if threshold_scheduler is None:
            threshold_scheduler = ExponentialThresholdScheduler(0.5, 1.0, 20)
        self._threshold_scheduler = threshold_scheduler

    @abstractmethod
    def train(self, subjects: Collection[Subject]):
        raise NotImplementedError()

    @abstractmethod
    def predict(self, subjects: Sequence[Subject]) -> Sequence[float]:
        raise NotImplementedError()

    def decide(self, subjects: Sequence[Subject]) -> List[Tuple[bool, float]]:
        scores = self.predict(subjects)
        return [
            (self._threshold_scheduler.decide(score, len(subject.posts) - 1), score)
            for subject, score in zip(subjects, scores)
        ]


class RandomBaseline(Model):
    def __init__(self, positive_ratio: float = 0.125, **kwargs):
        super().__init__(**kwargs)
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


class VocabularyBaseline(Model):
    def __init__(self, vocab_size: int = 200, min_count: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.min_count = min_count

    @classmethod
    def _tokenize(cls, text: str) -> List[str]:
        return re.findall(r"\w+", text)

    @classmethod
    def _word_counts(cls, subjects: Collection[Subject]):
        counts = defaultdict(int)
        for subject in subjects:
            for post in subject.posts:
                for word in cls._tokenize(post.text):
                    counts[word] += 1
        return counts

    def train(self, subjects: Collection[Subject]):
        neg_subjects = [subject for subject in subjects if subject.label is False]
        pos_subjects = [subject for subject in subjects if subject.label is True]
        neg_counts = self._word_counts(neg_subjects)
        neg_total = sum(neg_counts.values())
        pos_counts = self._word_counts(pos_subjects)
        pos_total = sum(pos_counts.values())
        neg_logfreqs = {
            word: np.log(neg_counts[word] / neg_total)
            for word in neg_counts
            if word in pos_counts and neg_counts[word] > 10 and pos_counts[word] > 10
        }
        pos_logfreqs = {
            word: np.log(pos_counts[word] / pos_total)
            for word in pos_counts
            if word in neg_counts and neg_counts[word] > 10 and pos_counts[word] > 10
        }
        self._pos_vocab = sorted(
            pos_logfreqs,
            key=lambda word: pos_logfreqs[word] - neg_logfreqs[word],
            reverse=True,
        )[: self.vocab_size]

    def predict(self, subjects: Sequence[Subject]) -> Sequence[float]:
        predictions = []
        for subject in subjects:
            last_post = self._tokenize(subject.posts[-1].text)
            num_matches = sum(word in self._pos_vocab for word in last_post)
            predictions.append(num_matches / (len(last_post) or 1) * 50)
        return predictions


class BertEmbeddingClassifier(Model):
    def __init__(
        self,
        layers: Collection[str] = (-4, -3, -2, -1),
        model_name: str = "bert-base-uncased",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.layers = layers
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        self._model = transformers.AutoModel.from_pretrained(
            model_name, output_hidden_states=True
        ).to(DEVICE)
        self._classifier = sklearn.linear_model.LogisticRegression(max_iter=10000)

    def _get_embeddings(self, text: str) -> torch.Tensor:
        tokens = self._tokenizer.encode(text, return_tensors="pt", truncation=True).to(
            DEVICE
        )
        with torch.no_grad():
            states = self._model(tokens).hidden_states
        embeddings = torch.stack([states[i] for i in self.layers]).sum(0).squeeze()
        return embeddings.cpu()

    def _encode_subject(self, subject: Subject) -> torch.Tensor:
        text = subject.posts[-1].text
        embeddings = self._get_embeddings(text)
        return embeddings.mean(0)  # TODO: Try other aggregations

    def train(self, subjects: Collection[Subject]):
        X = []
        y = []
        for subject in subjects:
            x = self._encode_subject(subject)
            X.append(x)
            y.append(float(subject.label))
        self._classifier.fit(torch.stack(X), y)

    def predict(self, subjects: Sequence[Subject]) -> Sequence[float]:
        X = torch.stack([self._encode_subject(subject) for subject in subjects])
        y_pred = self._classifier.predict(X)
        return y_pred


def save(model: Model, filename: str):
    with open(filename, "wb") as f:
        pickle.dump(model, f)


def load(filename: str) -> Model:
    with open(filename, "rb") as f:
        model = pickle.load(f)
    return model
