from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from xml.etree import ElementTree as ET


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


def parse_subject(filename: str) -> Subject:
    individual = ET.parse(filename)
    subject_id = individual.findtext("ID")
    posts = []
    for writing in individual.iterfind("WRITING"):
        title = writing.findtext("TITLE").strip()
        date = datetime.fromisoformat(writing.findtext("DATE").strip())
        text = writing.findtext("TEXT").strip()
        posts.append(Post(title, date, text))
    posts.sort(key=lambda post: post.date)
    label = Path(filename).parent.name
    assert label in ["neg", "pos"]
    return Subject(subject_id, posts, label == "pos")
