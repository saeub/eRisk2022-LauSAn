from abc import ABC, abstractmethod
from typing import Collection, Iterator, List, Tuple

from data import Subject


class Preprocessing(ABC):
    @abstractmethod
    def preprocess_for_training(
        self, subjects: Collection[Subject]
    ) -> Iterator[Tuple[str, bool]]:
        raise NotImplementedError()

    @abstractmethod
    def preprocess_for_prediction(self, subject: Subject) -> str:
        raise NotImplementedError()


class LatestPostsPreprocessing(Preprocessing):
    """Concatenate the most recent `n` posts in reverse order (most recent first)."""

    def __init__(self, n: int = 5):
        self.n = n

    def preprocess_for_training(
        self, subjects: Collection[Subject]
    ) -> Iterator[Tuple[str, bool]]:
        for subject in subjects:
            for i in range(len(subject.posts)):
                start = max(i - self.n + 1, 0)
                end = i + 1
                posts = subject.posts[start:end]
                yield " ".join(
                    reversed([post.title + " " + post.text for post in posts])
                ), subject.label

    def preprocess_for_prediction(self, subject: Subject) -> str:
        posts = subject.posts[-self.n :]
        text = " ".join(reversed([post.title + " " + post.text for post in posts]))
        return text


class AugmentedPreprocessing(Preprocessing):
    """
    Generate training samples by concatenating varying numbers of consecutive posts.
    Use the entire post history for prediction.
    :param numbers_to_concatinate: a list of integers that determines how many posts should be concatinated
     e.g. [2, 3]-> pairs of consecutive posts are concatinated and three consecutive posts are concatinated (data augmentation)
    :param overlap: a list of strings (posts by one subject)
    :param max_len: max number of words (seperated by whitespace) to be concatinated in preprocessing.
    This is an approximation on the maximum input length of the model, depends on the model! Used to avoid creating
    strings that are too long and will be truncated by the model --> this would result in identical data. E.g. 512 for BERT or 4096
    """
    def __init__(self, numbers_to_concatenate: List[int] = [2, 3, 4, 10, 20, 30, 40, 50], overlap: int = 0, max_len: int = 512):
        self.numbers_to_concatenate = numbers_to_concatenate
        self.overlap = overlap
        self.max_len = max_len

    def preprocess_for_training(
        self, subjects: Collection[Subject]
    ) -> Iterator[Tuple[str, bool]]:
        for subject in subjects:
            labels, texts = self._prepare_subject_data(subject)
            for text, label in zip(texts, labels):
                yield text, label

    def preprocess_for_prediction(self, subject: Subject) -> str:
        text = ""
        for i in reversed(range(len(subject.posts))):
            title_w_text = subject.posts[i].title + " " + subject.posts[i].text + " "
            text += title_w_text
        text = " ".join(text.split(" ")[:500])
        return text

    def _merge_posts(self, posts, n: int) -> List[str]:
        """
        Takes a list of strings (list of all posts by one subject) and merges strings
        in the list according to the specifications from the parameter n. The strings are
        merged in reverse order so that the oldest post is to the right and the newest
        post is to the left.
        :param posts: a list of strings (posts by one subject)
        :param number: the number of strings that should get merged into one string, this is taken by iterating over the
        list self.numbers_to_concatinate. It is specified if it should merge 2 posts or 3 posts etc.
        """

        merged_posts = []
        step = n - self.overlap
        for i in range(0, len(posts) - 1, step):
            # put the number of required sentences in a list
            count = 0  # repeat while loop as many times as the number of sentences we want to concatenate
            step2 = 0  # counter so it knows which sentence to pick next
            merged_sentence = (
                []
            )  # list for required sentences that need to be merged together

            while (
                count < n
            ):  # for as many times as the number of sentences we want to concatenate
                try:
                    sentence = posts[i + step2]
                    count += 1  # make one more iteration if the number of required sentence hasn't been reached yet
                    step2 += 1  # take one sentence to the right next time

                    merged_sentence.append(sentence)
                except IndexError:
                    break

            # limit augmentation to maximum input length of the model in order to avoid unnecessary concatenations which
            # will be duplicates after truncation
            if len(merged_sentence) == n:
                merged_sentence.reverse()  # newest post on the left (will be truncated on the right)
                merged_sent_str = " ".join(merged_sentence)
                if len(merged_sent_str.split()) <= self.max_len:
                    merged_posts.append(merged_sent_str)

        return merged_posts

    def _data_augmentation(self, subj_texts) -> List[str]:
        """
        Applies the _merge_posts function which merges e.g. 2 posts, 3 posts, etc. to all specified numbers in self.numbers_to_concatinate
        The newest post is always at the beginning of the string, the oldest at the end.
        :subj_texts: a list of strings (posts by one subject)
        """

        augmented_data = []

        # current post only (no history)
        for post in subj_texts:
            augmented_data.append(post)

        # current post + n posts of history
        for n in self.numbers_to_concatenate:
            # TODO: try out if it works better with an overlap (e.g. overlap 10% of n --> more data)
            for s in self._merge_posts(self, subj_texts, n): #TODO n
                augmented_data.append(s)

        return augmented_data

    def _prepare_subject_data(self, subject):
        """Takes a parsed subject, merges titles with texts and applies data augmentation to the posts:
        - list of labels of the same length as the list of augmented posts data
        - augmented data: list of merged posts
        :param subject: a parsed subject
        """

        # concatenate title and text -> new text
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
        augmented_texts = self._data_augmentation(self, subject_texts) #todo

        # get list with labels which is as long as augmented text list
        labels = [subject.label] * len(augmented_texts)

        return labels, augmented_texts
