from abc import ABC, abstractmethod
from typing import Collection, Iterator, List, Sequence, Tuple

from data import Subject


class Preprocessing(ABC):
    @abstractmethod
    def preprocess_for_training(
        self, subjects: Collection[Subject]
    ) -> Iterator[Tuple[str, bool]]:
        raise NotImplementedError()

    @abstractmethod
    def preprocess_for_prediction(self, subjects: Sequence[Subject]) -> Iterator[str]:
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

    def preprocess_for_prediction(self, subjects: Sequence[Subject]) -> Iterator[str]:
        for subject in subjects:
            posts = subject.posts[-self.n :]
            text = " ".join(reversed([post.title + " " + post.text for post in posts]))
            yield text, subject.label


class AugmentedPreprocessing(Preprocessing):
    """
    Generate training samples by concatenating varying numbers of consecutive posts.
    Use the entire post history for prediction.
    """

    def preprocess_for_training(
        self, subjects: Collection[Subject]
    ) -> Iterator[Tuple[str, bool]]:
        for subject in subjects:
            labels, texts = self._prepare_subject_data(
                subject, [2, 3, 4, 10, 20, 30, 40, 50], 0, 512
            )  # todo change 512 text len to variable
            for text, label in zip(texts, labels):
                yield text, label

    def preprocess_for_prediction(self, subject: Subject) -> str:
        text = ""
        for i in reversed(range(len(subject.posts))):
            title_w_text = subject.posts[i].title + " " + subject.posts[i].text + " "
            text += title_w_text
        text = " ".join(text.split(" ")[:500])
        return text

    def _merge_posts(self, posts, number: int, overlap: int, max_len: int) -> List[str]:
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
            count = 0  # repeat while loop as many times as the number of sentences we want to concatenate
            step2 = 0  # counter so it knows which sentence to pick next
            merged_sentence = (
                []
            )  # list for required sentences that need to be merged together

            while (
                count < number
            ):  # for as many times as the number of sentences we want to concatenate
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

    def _data_augmentation(
        self, posts, numbers_concat: List[int], overlap: int, max_len: int
    ) -> List[str]:
        """
        Function to augment the training and validation data.
        Takes a list of strings and returns concatenations of 2 posts, 3 posts, etc.
        The newest post is always at the beginning of the string, the oldest at the end.
        :param posts: a list of strings (posts by one subject)
        :param numbers_concat: list of integers that determines how many strings should be concatenated.
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
            for s in self._merge_posts(posts, n, overlap, max_len):
                augmented_data.append(s)

        return augmented_data

    def _prepare_subject_data(self, subject, numbers_to_concatenate, overlap, max_len):
        """Takes a filename for a subject and returns two lists:
        - list of labels of the same length as the list of augmented posts data
        - augmented data: list of merged posts
        :param filename: xml file for a subject
        :param numb_conc: list with numbers that determine how many posts of a subject should be concatenated.
        :param overlap: 0 if no overlap, 1 if 1 string overlap etc.
        :param max_len: maximal input length for model (e.g. 512 or 4096)
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
        augmented_texts = self._data_augmentation(
            subject_texts, numbers_to_concatenate, overlap, max_len
        )

        # get list with labels which is as long as augmented text list
        labels = [subject.label] * len(augmented_texts)

        return labels, augmented_texts
