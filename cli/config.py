from pathlib import Path

CACHE_DIR_PATH: Path = Path("cache")
INDEX_CACHE_PATH: Path = CACHE_DIR_PATH / "index.pkl"
DOCMAP_CACHE_PATH: Path = CACHE_DIR_PATH / "docmap.pkl"

BM25_K1: float = 1.5
BM25_B: float = 0.75

SEMANTIC_SEARCH_MODEL = "all-MiniLM-L6-v2"
