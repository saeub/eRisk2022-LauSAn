import pickle
import random
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Collection, Dict, List, Optional, Sequence, Tuple

import numpy as np
import sklearn.feature_extraction
import sklearn.linear_model
import sklearn.naive_bayes
import sklearn.neighbors
import sklearn.svm
import torch
import transformers
from tqdm import tqdm

import evaluation
from data import Subject
from log import logger
from preprocessing import (
    AugmentedPreprocessing,
    LatestPostsPreprocessing,
    Preprocessing,
)
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
        run: Optional[evaluation.Run] = None,
    ) -> evaluation.Run:
        """
        Optimize the threshold scheduler's parameters.

        This works by predicting scores for training subjects as if it was a real run,
        and then letting the ThresholdScheduler find the best parameters based on those
        scores and modify itself in-place.

        If a `run` is given, it will be used to perform grid search. Otherwise, a run
        is predicted first, and returned in the end to reuse with other metrics.
        """
        if run is None:
            logger.info(f"({self.__class__.__name__}) Predicting run...")
            subjects = list(subjects)
            if sample is not None:
                random.shuffle(subjects)
                subjects = subjects[:sample]
            run_subjects = [
                Subject(subject.id, [], subject.label) for subject in subjects
            ]
            run = {subject: [] for subject in run_subjects}

            num_posts_done = 0
            max_num_posts = max(len(subject.posts) for subject in subjects)
            progress = tqdm(total=max_num_posts)
            while num_posts_done < max_num_posts:
                # Copy over posts from `subjects` to `run_subjects` one by one
                for subject, run_subject in zip(subjects, run_subjects):
                    if len(subject.posts) > len(run_subject.posts):
                        run_subject.posts.append(subject.posts[len(run_subject.posts)])

                # Predict using the post histories up to this point, add scores to `run`
                run_subjects_to_predict = [
                    subject
                    for subject in run_subjects
                    if len(subject.posts) > len(run[subject])
                ]
                decisions = self.decide(run_subjects_to_predict)
                for subject, decision in zip(run_subjects_to_predict, decisions):
                    run[subject].append(decision)

                num_posts_done += 1
                progress.update()
            progress.refresh()

        logger.info(
            f"({self.__class__.__name__}) Performing grid search to optimize {metric}..."
        )
        self.threshold_scheduler.grid_search(
            self.threshold_scheduler_grid_search_parameters(),
            run,
            metric,
            minimize=minimize,
        )
        return run

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
        self._random = random.Random(42)

    def train(self, subjects: Collection[Subject]):
        pass

    def predict(self, subjects: Sequence[Subject]) -> Sequence[float]:
        predictions = []
        for subject in subjects:
            predictions.append(
                self.subject_predictions.setdefault(subject.id, self._random.random())
            )
        return predictions


class VocabularyBaseline(Model):
    def __init__(self, vocab_size: int = 200, min_count: int = 10):
        super().__init__(threshold_scheduler=ExponentialThresholdScheduler(0.1, 0.1, 1))
        self.vocab_size = vocab_size
        self.min_count = min_count

    def threshold_scheduler_grid_search_parameters(self) -> Dict[str, Collection[Any]]:
        return {
            "start_threshold": np.arange(0, 5, 0.5),
            "target_threshold": np.arange(1, 7, 0.5),
            "time_constant": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40, 50],
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
            predictions.append(num_matches / (len(last_post) or 1) * 100)
        return predictions


class NBClassifier(Model):
    def __init__(self, max_features: int = 1000):
        super().__init__(ExponentialThresholdScheduler(0.1, 0.5, 10))
        self._classifier = sklearn.naive_bayes.MultinomialNB()
        self._vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(
            max_features=max_features
        )

    def threshold_scheduler_grid_search_parameters(self) -> Dict[str, Collection[Any]]:
        return {
            "start_threshold": np.arange(0.0, 0.5, 0.1),
            "target_threshold": np.arange(0.0, 0.7, 0.1),
            "time_constant": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40, 50],
        }

    def train(self, subjects: Collection[Subject]):
        logger.info(f"({self.__class__.__name__}) Vectorizing posts...")
        X_texts = []
        y = []
        for subject in subjects:
            for post in subject.posts:
                X_texts.append(post.title + " " + post.text)
                y.append(int(subject.label))
        X = self._vectorizer.fit_transform(X_texts).toarray()
        logger.info(f"({self.__class__.__name__}) Fitting classifier...")
        self._classifier.fit(X, y)

    def predict(self, subjects: Sequence[Subject]) -> Sequence[float]:
        X_texts = []
        for subject in subjects:
            post = subject.posts[-1]
            X_texts.append(post.title + " " + post.text)
        X = self._vectorizer.transform(X_texts).toarray()
        y_pred = self._classifier.predict_proba(X)[:, 1]
        return y_pred


class BertEmbeddingClassifier(Model):
    def __init__(
        self,
        checkpoint: str = "bert-base-uncased",
        layers: Collection[str] = (-4, -3, -2, -1),
        classifier: str = "logistic_regression",
        preprocessing: str = "simple",
    ):
        super().__init__(ExponentialThresholdScheduler(0, 2, 10))
        self.layers = layers
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint)
        self._model = transformers.AutoModel.from_pretrained(
            checkpoint, output_hidden_states=True
        ).to(DEVICE)

        if classifier == "logistic_regression":
            self._classifier = sklearn.linear_model.LogisticRegression(max_iter=10000)
        elif classifier == "svm":
            self._classifier = sklearn.svm.SVC()
        elif classifier == "knn":
            self._classifier = sklearn.neighbors.KNeighborsClassifier()
        else:
            raise ValueError(f"Invalid classifier type {classifier}")

        if preprocessing == "simple":
            self._preprocessing = LatestPostsPreprocessing(1)
        elif preprocessing == "history":
            self._preprocessing = LatestPostsPreprocessing(5)
        elif preprocessing == "augmented":
            self._preprocessing = AugmentedPreprocessing()
        else:
            raise ValueError(f"Invalid preprocessing type {preprocessing}")

    def threshold_scheduler_grid_search_parameters(self) -> Dict[str, Collection[Any]]:
        return {
            "wait": range(5),
            "start_threshold": np.arange(-5, 5, 1),
            "target_threshold": np.arange(-5, 5, 1),
            "time_constant": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40, 50],
        }

    def _encode_text(self, text: str) -> torch.Tensor:
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
        for text, label in tqdm(self._preprocessing.preprocess_for_training(subjects)):
            x = self._encode_text(text)
            X.append(x)
            y.append(int(label))
        logger.info(f"({self.__class__.__name__}) Fitting classifier...")
        self._classifier.fit(torch.stack(X), y)

    def predict(self, subjects: Sequence[Subject]) -> Sequence[float]:
        X = []
        for subject in subjects:
            x = self._encode_posts(subject.posts)
            X.append(x)
        y_pred = self._classifier.decision_function(torch.stack(X))
        return y_pred


class TransformerDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        subjects: Collection[Subject],
        preprocessing: Preprocessing,
        tokenizer,
    ):
        self._items = []
        for text, label in preprocessing.preprocess_for_training(subjects):
            item = tokenizer(text, truncation=True)
            item["label"] = int(label)
            self._items.append(item)

    def __getitem__(self, index):
        return self._items[index]

    def __len__(self):
        return len(self._items)


class Transformer(Model):
    def __init__(
        self,
        checkpoint: str = "distilbert-base-uncased",
        preprocessing: str = "simple",
    ):
        super().__init__(ExponentialThresholdScheduler(0.3, 0.8, 20))
        self._checkpoint = checkpoint
        self._tokenizer = transformers.DistilBertTokenizerFast.from_pretrained(
            checkpoint
        )
        self._model = transformers.DistilBertForSequenceClassification.from_pretrained(
            checkpoint, num_labels=2
        )

        if preprocessing == "simple":
            self._preprocessing = LatestPostsPreprocessing(1)
        elif preprocessing == "history":
            self._preprocessing = LatestPostsPreprocessing(5)
        elif preprocessing == "augmented":
            self._preprocessing = AugmentedPreprocessing()
        else:
            raise ValueError(f"Invalid preprocessing type {preprocessing}")

    def threshold_scheduler_grid_search_parameters(self) -> Dict[str, Collection[Any]]:
        return {
            "start_threshold": np.arange(0.2, 1.01, 0.05),
            "target_threshold": np.arange(0.5, 1.01, 0.05),
            "time_constant": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40, 50],
        }

    def train(self, subjects: Collection[Subject]):
        logger.info(f"({self.__class__.__name__}) Preprocessing data...")
        dataset = TransformerDataset(
            list(subjects), self._preprocessing, self._tokenizer
        )
        trainer = transformers.Trainer(
            model=self._model,
            args=transformers.TrainingArguments(
                output_dir=f"./checkpoints-{self._checkpoint}",
                save_total_limit=3,
                num_train_epochs=3,
                per_device_train_batch_size=16,
                logging_steps=500,
                report_to="none",
            ),
            train_dataset=dataset,
            data_collator=transformers.DataCollatorWithPadding(self._tokenizer),
        )
        logger.info(f"({self.__class__.__name__}) Finetuning transformer...")
        trainer.train()

    def predict(self, subjects: Sequence[Subject]) -> Sequence[float]:
        scores = []
        for subject in subjects:
            text = self._preprocessing.preprocess_for_prediction(subject)
            item = self._tokenizer(text, truncation=True, return_tensors="pt")
            logits = self._model(item.input_ids.to(DEVICE)).logits
            score = float(torch.softmax(logits, 1)[0, 1])
            scores.append(score)
        return scores


class Ensemble(Model):
    def __init__(self, model_filenames: List[str]):
        super().__init__(ExponentialThresholdScheduler(0.5, 0.5, 1))
        self.model_filenames = model_filenames
        self._models = None

    def threshold_scheduler_grid_search_parameters(self) -> Dict[str, Collection[Any]]:
        first_model = load(self.model_filenames[0])
        assert isinstance(
            first_model.threshold_scheduler, ExponentialThresholdScheduler
        ), (
            "Threshold scheduler grid search for ensemble models currently only works "
            "if the first model uses an `ExponentialThresholdScheduler`."
        )
        return first_model.threshold_scheduler_grid_search_parameters()

    def train(self, subjects: Collection[Subject]):
        pass

    def predict(self, subjects: Sequence[Subject]) -> Sequence[float]:
        if self._models is None:
            self._models = [load(filename) for filename in self.model_filenames]
        model_scores = [model.predict(subjects) for model in self._models]
        mean_scores = [np.mean(subject_scores) for subject_scores in zip(*model_scores)]
        return mean_scores


def save(model: Model, filename: str):
    with open(filename, "wb") as f:
        pickle.dump(model, f)


def load(filename: str) -> Model:
    with open(filename, "rb") as f:
        model = pickle.load(f)
    return model
