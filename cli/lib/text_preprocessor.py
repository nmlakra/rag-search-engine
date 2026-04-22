import string
from typing import List
from pathlib import Path

STOP_WORD_FILE_PATH = Path("data/stopwords.txt")


def load_stopwords(fp: Path) -> List[str]:
    with open(fp) as f:
        lines = [l.strip() for l in f.readlines()]
        return lines


def remove_punctuation(text: str) -> str:
    translation_table = str.maketrans("", "", string.punctuation)
    res = text.translate(translation_table)
    return res


def tokenizer(text: str) -> List[str]:
    tokens = [text]
    if len(text) > 1:
        tokens = [_ for _ in text.split() if _]
    return tokens


def remove_stopwords(tokens: List[str], stopwords: List[str]) -> List[str]:
    filtered_tokens = [token for token in tokens if token not in stopwords]
    return filtered_tokens


def text_preprocessor(text: str) -> List[str]:
    text_lower = text.lower()
    text_depunctuated = remove_punctuation(text_lower)
    text_tokenized = tokenizer(text_depunctuated)

    stopwords = load_stopwords(STOP_WORD_FILE_PATH)
    text_stopwords_filterd = remove_stopwords(text_tokenized, stopwords)

    return text_stopwords_filterd
