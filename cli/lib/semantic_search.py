from sentence_transformers import SentenceTransformer
import numpy as np
from config import CACHE_DIR_PATH
from typing import List, Dict, Tuple
from pathlib import Path
from typing import TypedDict


class SemanticSearchResult(TypedDict):
    score: float
    title: str
    description: str


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    # Converting from numpy.float64 to native Python float for better readability in results
    return float(dot_product / (norm1 * norm2))


def verify_model(model_name: str) -> None:
    try:
        semantic_search = SemanticSearch(model_name)
    except Exception as e:
        raise Exception(f"Error loading model '{model_name}': {e}")

    print(f"Model loaded: {semantic_search.model}")
    print(f"Max sequence length: {semantic_search.model.max_seq_length}")


class SemanticSearch:
    """Class to handle semantic search operations using SentenceTransformer models."""

    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
        self.documents = None
        self.embeddings = None
        self.document_map = {}

        self.cache_dir = CACHE_DIR_PATH
        self.embeddings_cache_path = self.cache_dir / "embeddings.npy"

    def generate_embeddings(self, text: str) -> np.ndarray:
        text = text.strip()
        if not text:
            raise ValueError("Input text cannot be empty.")
        text_embedding = self.model.encode([text])
        return text_embedding

    def build_embeddings(self, documents: List[Dict]) -> np.ndarray:
        """Build embeddings for the provided documents and cache them."""
        self.documents = documents
        movies = []
        for doc in documents:
            self.document_map[doc["id"]] = doc
            doc_title = doc.get("title", "")
            doc_description = doc.get("description", "")
            if not doc_title or not doc_description:
                print(
                    f"Warning: Document with id {doc['id']} is missing title or description. Skipping."
                )
                continue
            movie = f"{doc_title}: {doc_description}"
            movies.append(movie)

        self.embeddings = self.model.encode(movies)
        self._save_embeddings(self.embeddings_cache_path)
        return self.embeddings

    def _save_embeddings(self, file_path: Path) -> None:
        """Save embeddings to a file."""
        if self.embeddings is None:
            raise ValueError("Embeddings have not been generated yet.")
        np.save(file_path, self.embeddings)

    def load_or_build_embeddings(self, documents: List[Dict]) -> np.ndarray:
        """Load cached embeddings if available, otherwise build new embeddings and cache them."""
        if self.embeddings_cache_path.exists():
            self.documents = documents
            self.documents = {doc["id"]: doc for doc in documents}
            print("Loading cached embeddings...")
            self.embeddings = np.load(self.embeddings_cache_path)
            if self.embeddings.shape[0] != len(self.documents):
                raise ValueError(
                    "Cached embeddings count does not match the number of documents. Please rebuild the embeddings."
                )
        else:
            print("No cached embeddings found. Building new embeddings...")
            self.build_embeddings(documents)

        if self.embeddings is None:
            raise ValueError("Failed to load or build embeddings.")

        return self.embeddings

    def search(self, query: str, limit: int) -> List[SemanticSearchResult]:
        if self.embeddings is None:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )

        query_embeddings = self.generate_embeddings(query)
        query_similarity_scores = [
            cosine_similarity(query_embeddings, doc_embeddings)
            for doc_embeddings in self.embeddings
        ]

        if not self.documents:
            raise Exception(
                "No documents available for search. Please load or build embeddings first."
            )

        doc_scores = [
            (similarity_score, document)
            for similarity_score, document in zip(
                query_similarity_scores, self.documents.values()
            )
        ]

        doc_scores.sort(key=lambda x: x[0], reverse=True)
        semantic_search_results = self._format_semantic_search_results(
            doc_scores, limit
        )

        return semantic_search_results

    @staticmethod
    def _format_semantic_search_results(
        doc_scores: List[Tuple[float, Dict]], limit: int
    ) -> List[SemanticSearchResult]:
        results = []
        for score, document in doc_scores[:limit]:
            results.append(
                {
                    "score": score,
                    "title": document["title"],
                    "description": document["description"],
                }
            )

        return results
