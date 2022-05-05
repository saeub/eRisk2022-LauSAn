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
        history: int = 1,
        classifier: str = "logistic_regression",
    ):
        super().__init__(ExponentialThresholdScheduler(0, 2, 10))
        self.layers = layers
        self.history = history
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

    def threshold_scheduler_grid_search_parameters(self) -> Dict[str, Collection[Any]]:
        return {
            "wait": range(5),
            "start_threshold": np.arange(-5, 5, 1),
            "target_threshold": np.arange(-5, 5, 1),
            "time_constant": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40, 50],
        }

    def _encode_posts(self, posts: Sequence[Post]) -> torch.Tensor:
        text = " ".join(reversed([post.title + " " + post.text for post in posts[-self.history :]]))
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
            for i in range(len(subject.posts)):
                x = self._encode_posts(subject.posts[: i + 1])
                X.append(x)
                y.append(int(subject.label))
        logger.info(f"({self.__class__.__name__}) Fitting classifier...")
        self._classifier.fit(torch.stack(X), y)

    def predict(self, subjects: Sequence[Subject]) -> Sequence[float]:
        X = []
        for subject in subjects:
            x = self._encode_posts(subject.posts)
            X.append(x)
        y_pred = self._classifier.decision_function(torch.stack(X))
        return y_pred


class TransformersDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        subjects: Sequence[Subject],
        tokenizer,
    ):
        # TODO: Concatenate final posts, truncate from start
        self._texts = [
            tokenizer(post.title + " " + post.text, truncation=True)
            for subject in subjects
            for post in subject.posts
        ]
        self._labels = [
            int(subject.label) for subject in subjects for _ in subject.posts
        ]

    def __getitem__(self, index):
        item = self._texts[index]
        item["labels"] = self._labels[index]
        return item

    def __len__(self):
        return len(self._texts)


class TransformersConcatinatedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        subjects: Sequence[Subject],
        tokenizer,
    ):

        self._texts = []
        self._labels = []

        for subject in subjects:
            labels, texts = self.prepare_subject_data(
                subject, [2, 3, 4, 10, 20, 30, 40, 50], 0, 512
            )  # todo change 512 text len to variable
            self._texts.extend([tokenizer(t, truncation=True) for t in texts])
            self._labels.extend(labels)

    def __getitem__(self, index):
        item = self._texts[index]
        item["labels"] = self._labels[index]
        return item

    def __len__(self):
        return len(self._texts)

    def merge_posts(self, posts, number: int, overlap: int, max_len: int) -> List[str]:
        """
        Takes a list of strings (list of all posts by one subject) and merges strings
        in the list according to the specifications from the parameters. The strings are
        merged in reverse order so that the oldest post is to the right and the newest
        post is to the left.
        :param posts: a list of strings (posts by one subject)
        :param number: the number of strings that should get merged into one string,
        must be > 0 (e.g. number = 2 will always merge two strings together)
        :param overlap: 0 if no overlap, 1 if 1 string overlap etc.
        :param max_len: maximal input length for model (e.g. 512 or 4096)
        """

        merged_posts = []
        step = number - overlap
        for i in range(0, len(posts) - 1, step):
            # put the number of required sentences in a list
            count = 0  # repeat while loop as many times as the number of sentences we want to concatinate
            step2 = 0  # counter so it knows which sentence to pick next
            merged_sentence = (
                []
            )  # list for required sentences that need to be merged together

            while (
                count < number
            ):  # for as many times as the number of sentences we want to concatinate
                try:
                    sentence = posts[i + step2]
                    count += 1  # make one more iteration if the number of required sentence hasn't been reached yet
                    step2 += 1  # take one sentence to the right next time

                    merged_sentence.append(sentence)
                except IndexError:
                    break

            # nur sätze nehmen, bei denen es aufgeht (=duplikate vermeiden) und die ins modell passen
            if len(merged_sentence) == number:
                merged_sentence.reverse()  # newest post on the left (will be truncated on the right)
                merged_sent_str = " ".join(merged_sentence)
                if len(merged_sent_str.split()) <= max_len:
                    merged_posts.append(merged_sent_str)

        return merged_posts

    def data_augmentation(
        self, posts, numbers_concat: List[int], overlap: int, max_len: int
    ) -> List[str]:
        """
        Function to augment the training and validation data.
        Takes a list of strings and returns concatinations of 2 posts, 3 posts, etc.
        The newest post is always at the beginning of the string, the oldest at the end.
        :param posts: a list of strings (posts by one subject)
        :param numbers_concat: list of integers that determines how many strings should be concatinated.
        :param overlap: 0 if no overlap, 1 if 1 string overlap etc.
        :param max_len: maximal input length for model (e.g. 512 or 4096)
        """

        augmented_data = []

        # current post only (no history)
        for post in posts:
            augmented_data.append(post)

        # current post + n posts of history
        for n in numbers_concat:
            # TODO: try out if it works better with an overlap (e.g. overlap 10% of n --> more data)
            for s in self.merge_posts(posts, n, overlap, max_len):
                augmented_data.append(s)

        return augmented_data

    def prepare_subject_data(self, subject, numbers_to_concatinate, overlap, max_len):
        """Takes a filename for a subject and returns two lists:
        - list of labels of the same length as the list of augmented posts data
        - augmented data: list of merged posts
        :param filename: xml file for a subject
        :param numb_conc: list with numbers that determine how many posts of a subject should be concatinated.
        :param overlap: 0 if no overlap, 1 if 1 string overlap etc.
        :param max_len: maximal input length for model (e.g. 512 or 4096)
        """

        subject_id = subject.id

        # get subject label
        if subject.label == True:
            subject_label = 1
        elif subject.label == False:
            subject_label = 0

        # concatinate title and text -> new text
        subject_texts = []
        for post in subject.posts:
            if post.text != "" and post.title != "":
                text_title = post.title + " " + post.text
                subject_texts.append(text_title)
            elif post.text == "" and post.title != "":
                subject_texts.append(post.title)
            elif post.text != "" and post.title == "":
                subject_texts.append(post.text)
            else:
                pass

        # augment text
        augmented_texts = self.data_augmentation(
            subject_texts, numbers_to_concatinate, overlap, max_len
        )

        # get list with labels which is as long as augmented text list
        labels = [subject_label] * len(augmented_texts)

        return labels, augmented_texts


class WeightedLossTrainer(transformers.Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_function = torch.nn.CrossEntropyLoss(
            weight=torch.tensor([1.0, 8.0]).to(DEVICE)
        )
        loss = loss_function(
            logits.view(-1, self.model.config.num_labels), labels.view(-1)
        )
        return (loss, outputs) if return_outputs else loss


class Roberta(Model):
    def __init__(self, checkpoint: str = "roberta-base"):
        super().__init__(ExponentialThresholdScheduler(0.3, 0.8, 20))
        self._tokenizer = transformers.RobertaTokenizer.from_pretrained(checkpoint)
        self._model = transformers.RobertaForSequenceClassification.from_pretrained(
            checkpoint, num_labels=2
        )

    def threshold_scheduler_grid_search_parameters(self) -> Dict[str, Collection[Any]]:
        return {
            "start_threshold": np.arange(0.2, 1.01, 0.05),
            "target_threshold": np.arange(0.5, 1.01, 0.05),
            "time_constant": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40, 50],
        }

    def train(self, subjects: Collection[Subject]):
        dataset = TransformersDataset(list(subjects), self._tokenizer)
        trainer = transformers.Trainer(
            model=self._model,
            args=transformers.TrainingArguments(
                output_dir="./roberta-checkpoints",
                save_total_limit=1,
                num_train_epochs=3,
            ),
            train_dataset=dataset,
            data_collator=transformers.DataCollatorWithPadding(self._tokenizer),
        )
        trainer.train()

    def predict(self, subjects: Sequence[Subject]) -> Sequence[float]:
        scores = []
        for subject in subjects:
            post = subject.posts[-1]
            item = self._tokenizer(
                post.title + " " + post.text, truncation=True, return_tensors="pt"
            )
            logits = self._model(item.input_ids.to(DEVICE)).logits
            score = float(torch.softmax(logits, 1)[0, 1])
            scores.append(score)
        return scores


class Electra(Model):
    def __init__(self, checkpoint: str = "google/electra-base-discriminator"):
        super().__init__(ExponentialThresholdScheduler(0.3, 0.8, 20))
        self._tokenizer = transformers.ElectraTokenizer.from_pretrained(checkpoint)
        self._model = transformers.ElectraForSequenceClassification.from_pretrained(
            checkpoint, num_labels=2
        )

    def threshold_scheduler_grid_search_parameters(self) -> Dict[str, Collection[Any]]:
        return {
            "start_threshold": np.arange(0.2, 1.01, 0.05),
            "target_threshold": np.arange(0.5, 1.01, 0.05),
            "time_constant": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40, 50],
        }

    def train(self, subjects: Collection[Subject]):
        dataset = TransformersDataset(list(subjects), self._tokenizer)
        trainer = transformers.Trainer(
            model=self._model,
            args=transformers.TrainingArguments(
                output_dir="./electra-checkpoints",
                save_total_limit=3,
                num_train_epochs=4,
                per_device_train_batch_size=12,
            ),
            train_dataset=dataset,
            data_collator=transformers.DataCollatorWithPadding(self._tokenizer),
        )
        trainer.train()

    def predict(self, subjects: Sequence[Subject]) -> Sequence[float]:
        scores = []
        for subject in subjects:
            post = subject.posts[-1]
            item = self._tokenizer(
                post.title + " " + post.text, truncation=True, return_tensors="pt"
            )
            logits = self._model(item.input_ids.to(DEVICE)).logits
            score = float(torch.softmax(logits, 1)[0, 1])
            scores.append(score)
        return scores


class Longformer(Model):
    def __init__(self, checkpoint: str = "allenai/longformer-base-4096"):
        super().__init__(ExponentialThresholdScheduler(0.3, 0.8, 20))
        self._tokenizer = transformers.LongformerTokenizerFast.from_pretrained(
            checkpoint
        )
        self._model = transformers.LongformerForSequenceClassification.from_pretrained(
            checkpoint, num_labels=2
        )
        # self._model.config.attention_window = 256

    def train(self, subjects: Collection[Subject]):
        dataset = TransformersConcatinatedDataset(list(subjects), self._tokenizer)
        trainer = transformers.Trainer(
            model=self._model,
            args=transformers.TrainingArguments(
                output_dir="./longformer-checkpoints",
                save_total_limit=3,
                num_train_epochs=3,
                per_device_train_batch_size=2,
                logging_steps=500,
                report_to=None,
            ),
            train_dataset=dataset,
            data_collator=transformers.DataCollatorWithPadding(self._tokenizer),
        )
        logger.info("Fitting Longformer classifier...")
        torch.cuda.empty_cache()
        trainer.train()

    def predict(self, subjects: Sequence[Subject]) -> Sequence[float]:
        scores = []
        for subject in subjects:
            concat_post = ""
            for i in reversed(range(len(subject.posts))):
                title_w_text = (
                    subject.posts[i].title + " " + subject.posts[i].text + " "
                )
                concat_post += title_w_text
            item = self._tokenizer(concat_post, truncation=True, return_tensors="pt")
            logits = self._model(item.input_ids.to(DEVICE)).logits
            score = float(torch.softmax(logits, 1)[0, 1])
            scores.append(score)
        return scores


class DistilBertConcatenated(Model):
    def __init__(self, checkpoint: str = "distilbert-base-uncased"):
        super().__init__(ExponentialThresholdScheduler(0.3, 0.8, 20))
        self._tokenizer = transformers.DistilBertTokenizerFast.from_pretrained(
            checkpoint
        )
        self._model = transformers.DistilBertForSequenceClassification.from_pretrained(
            checkpoint, num_labels=2
        )

    def threshold_scheduler_grid_search_parameters(self) -> Dict[str, Collection[Any]]:
        return {
            "start_threshold": np.arange(0.2, 1.01, 0.05),
            "target_threshold": np.arange(0.5, 1.01, 0.05),
            "time_constant": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40, 50],
        }

    def train(self, subjects: Collection[Subject]):
        dataset = TransformersConcatinatedDataset(list(subjects), self._tokenizer)
        trainer = transformers.Trainer(
            model=self._model,
            args=transformers.TrainingArguments(
                output_dir="./distilbert-concatenated-checkpoints",
                save_total_limit=3,
                num_train_epochs=3,
                per_device_train_batch_size=22,
                logging_steps=500,
                report_to=None,
            ),
            train_dataset=dataset,
            data_collator=transformers.DataCollatorWithPadding(self._tokenizer),
        )
        logger.info("Fitting DistillBert classifier...")
        trainer.train()

    def predict(self, subjects: Sequence[Subject]) -> Sequence[float]:
        scores = []
        for subject in subjects:
            concat_post = ""
            for i in reversed(range(len(subject.posts))):
                title_w_text = (
                    subject.posts[i].title + " " + subject.posts[i].text + " "
                )
                concat_post += title_w_text
            concat_post = " ".join(concat_post.split(" ")[:500])
            item = self._tokenizer(concat_post, truncation=True, return_tensors="pt")
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
