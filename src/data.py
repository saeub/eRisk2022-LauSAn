from dataclasses import dataclass
from datetime import datetime
from typing import List
from xml.etree import ElementTree as ET


@dataclass
class Post:
    title: str
    date: datetime
    info: str
    text: str


@dataclass
class Subject:
    id: str
    posts: List[Post]


def parse_subject(filename: str) -> Subject:
    individual = ET.parse(filename)
    id = individual.find("ID")
    posts = []
    for writing in individual.iterfind("WRITING"):
        title = writing.findtext("TITLE").strip()
        date = datetime.fromisoformat(writing.findtext("DATE").strip())
        info = writing.findtext("INFO").strip()
        text = writing.findtext("TEXT").strip()
        posts.append(Post(title, date, info, text))
    return Subject(id, posts)
