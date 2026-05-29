from typing import Dict, List, Tuple
import math
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
        self.doc_lengths = defaultdict(int)

        self.cache_path = cache_path
        self.index_cache_path = self.cache_path / "index.pkl"
        self.docmap_cache_path = self.cache_path / "docmap.pkl"
        self.tf_cache_path = self.cache_path / "term_frequencies.pkl"
        self.doc_lengths_cache_path = self.cache_path / "doc_lengths.pkl"

    def __add_document(self, doc_id: int, doc: Document) -> None:
        text = ""
        for val in asdict(doc).values():
            text += str(val) + " "
        tokens = text_preprocessor(text)
        for token in tokens:
            self.index[token].add(doc_id)
        self.term_frequencies[doc_id] = Counter(tokens)
        self.doc_lengths[doc_id] = len(tokens)

    def __get_avg_doc_length(self) -> float:
        if self.doc_lengths:
            n_docs = len(self.doc_lengths)
            total_length = sum(self.doc_lengths.values())

            return total_length / n_docs
        return 0.0

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

    def get_idf(self, term: str) -> float:
        token = text_preprocessor(term)[0]
        total_doc_count = len(self.docmap)
        term_match_doc_count = len(self.index[token])

        # IDF formula with smoothing
        # log((N + 1) / (n + 1))
        # N = total number of documents
        # n = number of documents containing the term
        idf_num = total_doc_count + 1
        idf_den = term_match_doc_count + 1
        idf = math.log(idf_num / idf_den)

        return idf

    def get_bm25_idf(self, term: str) -> float:
        token, *_ = text_preprocessor(term)
        if _:
            ValueError("More than one term present.")
        total_doc_count = len(self.docmap)
        term_match_doc_count = len(self.index[token])

        # BM25 IDF formula with smoothing
        # log((N - n + 0.5) / (n + 0.5))
        # N = total number of documents
        # n = number of documents containing the term
        idf_num = total_doc_count - term_match_doc_count + 0.5
        idf_den = term_match_doc_count + 0.5
        idf = math.log(idf_num / idf_den + 1)

        return idf

    def get_tf_idf(self, doc_id: int, term: str) -> float:
        tf = self.get_tf(doc_id, term)
        idf = self.get_idf(term)

        return tf * idf

    def __length_norm(self, doc_id: int, b: float) -> float:
        doc_length = self.doc_lengths[doc_id]
        avg_doc_length = self.__get_avg_doc_length()
        return (1 - b) + (b * (doc_length / avg_doc_length))

    def get_bm25_tf(self, doc_id: int, term: str, k1: float, b: float) -> float:
        length_norm = self.__length_norm(doc_id, b)
        tf = self.get_tf(doc_id, term)

        bm25_num = tf * (k1 + 1)
        bm25_den = tf + k1 * length_norm

        return bm25_num / bm25_den

    def bm25(self, doc_id: int, term: str, k1: float, b: float) -> float:
        bm25_tf = self.get_bm25_tf(doc_id, term, k1, b)
        bm25_idf = self.get_bm25_idf(term)

        return bm25_tf * bm25_idf

    def bm25_search(
        self, query: str, limit: int, k1: float, b: float
    ) -> List[Tuple[int, float]]:
        query_tokens = text_preprocessor(query)

        doc_ids = self.doc_lengths.keys()
        scores = {}
        for doc_id in doc_ids:
            doc_scores = []
            for token in query_tokens:
                doc_scores.append(self.bm25(doc_id, token, k1, b))
            scores[doc_id] = sum(doc_scores)

        top_docs = self._get_top_docs(scores, limit)

        return top_docs

    def _get_top_docs(
        self, doc_scores: Dict[int, float], limit: int
    ) -> List[Tuple[int, float]]:
        sorted_docs_scores = sorted(
            doc_scores.items(), key=lambda x: x[1], reverse=True
        )

        return sorted_docs_scores[:limit]

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

        with open(self.doc_lengths_cache_path, "wb") as doc_lengths_cache:
            pickle.dump(self.doc_lengths, doc_lengths_cache)

    def load(self) -> None:
        if os.path.exists(self.cache_path):
            with open(self.index_cache_path, "rb") as index_cache:
                self.index = pickle.load(index_cache)

            with open(self.docmap_cache_path, "rb") as docmap_cache:
                self.docmap = pickle.load(docmap_cache)

            with open(self.tf_cache_path, "rb") as tf_cache:
                self.term_frequencies = pickle.load(tf_cache)

            with open(self.doc_lengths_cache_path, "rb") as doc_lengths_cache:
                self.doc_lengths = pickle.load(doc_lengths_cache)
        else:
            raise FileNotFoundError("Provided cache file path not found.")
