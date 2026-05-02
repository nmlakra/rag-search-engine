from dataclasses import dataclass
from abc import ABC


class Document(ABC):
    pass


@dataclass
class Movie(Document):
    title: str
    description: str
