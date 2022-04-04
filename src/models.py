import pickle
import random
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Collection, Dict, List, Optional, Sequence, Tuple

import numpy as np
import sklearn.linear_model
import torch
import transformers
from tqdm import tqdm

import evaluation
from data import Post, Subject
from log import logger
from threshold_schedulers import (
    ConstantThresholdScheduler,
    ExponentialThresholdScheduler,
    ThresholdScheduler,
)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


class Model(ABC):
    def __init__(self, threshold_scheduler: ThresholdScheduler):
        if threshold_scheduler is None:
            threshold_scheduler = ConstantThresholdScheduler(0.0)
        self.threshold_scheduler = threshold_scheduler

    def optimize_threshold_scheduler(
        self,
        subjects: Collection[Subject],
        metric: evaluation.Metric,
        minimize: bool = False,
        sample: Optional[int] = None,
    ):
        """
        Optimize the threshold scheduler's parameters.

        This works by predicting scores for training subjects as if it was a real run,
        and then letting the ThresholdScheduler find the best parameters based on those
        scores and modify itself in-place.
        """
        logger.info(f"({self.__class__.__name__}) Predicting run...")
        subjects = list(subjects)
        if sample is not None:
            random.shuffle(subjects)
            subjects = subjects[:sample]
        run_subjects = [Subject(subject.id, [], subject.label) for subject in subjects]
        run = {subject: [] for subject in run_subjects}

        while True:
            any_posts_left = False
            # Copy over posts from `subjects` to `run_subjects` one by one
            for subject, run_subject in zip(subjects, run_subjects):
                if len(subject.posts) > len(run_subject.posts):
                    run_subject.posts.append(subject.posts[len(run_subject.posts)])
                    any_posts_left = True
            if not any_posts_left:
                break

            # Predict using the post histories up to this point, add scores to `run`
            run_subjects_to_predict = [
                subject
                for subject in run_subjects
                if len(subject.posts) > len(run[subject])
            ]
            decisions = self.decide(run_subjects_to_predict)
            for subject, decision in zip(run_subjects_to_predict, decisions):
                run[subject].append(decision)

        logger.info(f"({self.__class__.__name__}) Performing grid search...")
        self.threshold_scheduler.grid_search(
            self.threshold_scheduler_grid_search_parameters(),
            run,
            metric,
            minimize=minimize,
        )

    def threshold_scheduler_grid_search_parameters(self) -> Dict[str, Collection[Any]]:
        return {}

    @abstractmethod
    def train(self, subjects: Collection[Subject]):
        raise NotImplementedError()

    @abstractmethod
    def predict(self, subjects: Sequence[Subject]) -> Sequence[float]:
        raise NotImplementedError()

    def decide(self, subjects: Sequence[Subject]) -> List[Tuple[bool, float]]:
        scores = self.predict(subjects)
        return [
            (self.threshold_scheduler.decide(score, len(subject.posts) - 1), score)
            for subject, score in zip(subjects, scores)
        ]


class RandomBaseline(Model):
    def __init__(self, positive_ratio: float = 0.125):
        super().__init__(ConstantThresholdScheduler(1 - positive_ratio))
        self.positive_ratio = positive_ratio
        self.subject_predictions = {}

    def train(self, subjects: Collection[Subject]):
        pass

    def predict(self, subjects: Sequence[Subject]) -> Sequence[float]:
        predictions = []
        for subject in subjects:
            predictions.append(
                self.subject_predictions.setdefault(subject.id, random.random())
            )
        return predictions


class VocabularyBaseline(Model):
    def __init__(self, vocab_size: int = 200, min_count: int = 10):
        super().__init__(threshold_scheduler=ConstantThresholdScheduler(0.5))
        self.vocab_size = vocab_size
        self.min_count = min_count

    def threshold_scheduler_grid_search_parameters(self) -> Dict[str, Collection[Any]]:
        return {
            "threshold": np.arange(0, 1, 0.1),
        }

    @classmethod
    def _tokenize(cls, text: str) -> List[str]:
        return re.findall(r"\w+", text)

    @classmethod
    def _word_counts(cls, subjects: Collection[Subject]) -> Dict[str, int]:
        counts = defaultdict(int)
        for subject in subjects:
            for post in subject.posts:
                for word in cls._tokenize(post.title + " " + post.text):
                    counts[word] += 1
        return counts

    def train(self, subjects: Collection[Subject]):
        neg_subjects = [subject for subject in subjects if not subject.label]
        pos_subjects = [subject for subject in subjects if subject.label]
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
    ):
        super().__init__(ExponentialThresholdScheduler(0, 2, 10))
        self.layers = layers
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        self._model = transformers.AutoModel.from_pretrained(
            model_name, output_hidden_states=True
        ).to(DEVICE)
        self._classifier = sklearn.linear_model.LogisticRegression(max_iter=10000)

    def threshold_scheduler_grid_search_parameters(self) -> Dict[str, Collection[Any]]:
        return {
            "start_threshold": np.arange(-10, 5, 1),
            "target_threshold": np.arange(-5, 10, 1),
            "time_constant": np.arange(1, 10, 1),
        }

    def _encode_post(self, post: Post) -> torch.Tensor:
        text = post.title + " " + post.text
        tokens = self._tokenizer.encode(text, return_tensors="pt", truncation=True).to(
            DEVICE
        )
        with torch.no_grad():
            states = self._model(tokens).hidden_states
        embeddings = torch.stack([states[i] for i in self.layers]).sum(0).squeeze()
        return embeddings.mean(0).cpu()  # TODO: Try other aggregations

    def train(self, subjects: Collection[Subject]):
        logger.info(f"({self.__class__.__name__}) Encoding posts...")
        X = []
        y = []
        for subject in tqdm(subjects):
            for post in subject.posts:
                x = self._encode_post(post)
                X.append(x)
                y.append(float(subject.label))
        logger.info(f"({self.__class__.__name__}) Fitting classifier...")
        self._classifier.fit(torch.stack(X), y)

    def predict(self, subjects: Sequence[Subject]) -> Sequence[float]:
        X = []
        for subject in subjects:
            x = self._encode_post(subject.posts[-1])
            X.append(x)
        y_pred = self._classifier.decision_function(torch.stack(X))
        return y_pred


def save(model: Model, filename: str):
    with open(filename, "wb") as f:
        pickle.dump(model, f)


def load(filename: str) -> Model:
    with open(filename, "rb") as f:
        model = pickle.load(f)
    return model
