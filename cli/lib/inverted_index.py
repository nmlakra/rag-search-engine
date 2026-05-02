from typing import Dict, List
from collections import defaultdict, Counter
from lib.text_preprocessor import text_preprocessor
import os
from pathlib import Path
import pickle
from lib.document import Document, Movie
from dataclasses import asdict


class InvertedIndex:
    def __init__(self, cache_path: Path) -> None:
        self.index = defaultdict(set)
        self.docmap = defaultdict(set)
        self.term_frequencies = defaultdict(dict)

        self.cache_path = cache_path
        self.index_cache_path = self.cache_path / "index.pkl"
        self.docmap_cache_path = self.cache_path / "docmap.pkl"
        self.tf_cache_path = self.cache_path / "term_frequencies.pkl"

    def __add_document(self, doc_id: int, doc: Document) -> None:
        text = ""
        for val in asdict(doc).values():
            text += str(val) + " "
        tokens = text_preprocessor(text)
        for token in tokens:
            self.index[token].add(doc_id)
        self.term_frequencies[doc_id] = Counter(tokens)

    def get_documents(self, term: str) -> List[int]:
        return sorted(self.index[term])

    def get_tf(self, doc_id: int, term: str) -> int:
        token, *_ = text_preprocessor(term)
        if _:
            ValueError("More than one term present.")
        if doc_id in self.term_frequencies:
            doc_tf = self.term_frequencies[doc_id]
        else:
            print(self.term_frequencies)
            raise KeyError("Invalid document ID.")
        if token in doc_tf:
            return doc_tf[token]
        else:
            return 0

    def build(self, documents: Dict) -> None:
        for document in documents["movies"]:
            doc_id = document["id"]
            doc = Movie(title=document["title"], description=document["description"])
            self.docmap[doc_id] = doc
            self.__add_document(doc_id, doc)

    def save(self) -> None:
        os.makedirs(self.cache_path, exist_ok=True)

        with open(self.index_cache_path, "wb") as index_cache:
            pickle.dump(self.index, index_cache)

        with open(self.docmap_cache_path, "wb") as docmap_cache:
            pickle.dump(self.docmap, docmap_cache)

        with open(self.tf_cache_path, "wb") as tf_cache:
            pickle.dump(self.term_frequencies, tf_cache)

    def load(self) -> None:
        if os.path.exists(self.cache_path):
            with open(self.index_cache_path, "rb") as index_cache:
                self.index = pickle.load(index_cache)

            with open(self.docmap_cache_path, "rb") as docmap_cache:
                self.docmap = pickle.load(docmap_cache)

            with open(self.tf_cache_path, "rb") as tf_cache:
                self.term_frequencies = pickle.load(tf_cache)
        else:
            raise FileNotFoundError("Provided cache file path not found.")
