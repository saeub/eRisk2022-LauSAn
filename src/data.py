from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence
from xml.etree import ElementTree as ET

import dateutil.parser

from log import logger


@dataclass
class Post:
    title: str
    date: datetime
    text: str


@dataclass
class Subject:
    id: str
    posts: List[Post]
    label: Optional[bool]

    def __str__(self) -> str:
        return f"<Subject '{self.id}' ({self.label})>"

    def __hash__(self) -> int:
        return hash(self.id)


def parse_subject(filename: str) -> Subject:
    individual = ET.parse(filename)
    subject_id = individual.findtext("ID")
    posts = []
    for writing in individual.iterfind("WRITING"):
        title = writing.findtext("TITLE").strip()
        date = dateutil.parser.isoparse(writing.findtext("DATE").strip())
        text = writing.findtext("TEXT").strip()
        posts.append(Post(title, date, text))
    posts.sort(key=lambda post: post.date)
    label = Path(filename).parent.name
    assert label in ["neg", "pos"]
    return Subject(subject_id, posts, label == "pos")


def undersample_subjects(subjects: Sequence[Subject], ratio: float) -> List[Subject]:
    """Undersample negative subjects to achieve the given negative-to-positive ratio."""
    num_pos_subjects = num_neg_subjects = 0
    for subject in subjects:
        if subject.label:
            num_pos_subjects += 1
        else:
            num_neg_subjects += 1
    neg_ratio_to_delete = 1 - ratio * num_pos_subjects / num_neg_subjects
    undersampled_subjects = []
    i = 0
    for subject in subjects:
        if subject.label:
            # Keep all positive subjects
            undersampled_subjects.append(subject)
        else:
            i += neg_ratio_to_delete
            if i >= 1:
                # Delete some negative subjects according to ratio
                i -= 1
            else:
                # Keep remaining negative subjects
                undersampled_subjects.append(subject)

    num_neg_subjects_after = 0
    for subject in undersampled_subjects:
        if not subject.label:
            num_neg_subjects_after += 1
    logger.info(
        f"Undersampled {num_neg_subjects} to {num_neg_subjects_after} "
        "negative subjects."
    )

    return undersampled_subjects
