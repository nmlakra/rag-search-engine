from dataclasses import dataclass
from abc import ABC


@dataclass
class Document(ABC):
    pass


@dataclass
class Movie(Document):
    title: str
    description: str
