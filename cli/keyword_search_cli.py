import math
import argparse
import json
from typing import List
from lib.text_preprocessor import text_preprocessor
from lib.inverted_index import InvertedIndex
from config import CACHE_DIR_PATH


def inverted_search(query: str, indexer: InvertedIndex, search_limit: int = 5) -> List:
    query_tokens = text_preprocessor(query)
    search_results = []

    for query_token in query_tokens:
        search_results.extend(indexer.get_documents(query_token))

        if len(search_results) >= 5:
            break

    return [indexer.docmap[res].title for res in search_results[:5]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("build", help="Creates inverted index")

    tf_parser = subparsers.add_parser(
        "tf", help="Provides term frequencies for <document id> <term>"
    )
    tf_parser.add_argument("doc_id", help="Document id", type=int)
    tf_parser.add_argument("term", help="Term")

    tfidf_parser = subparsers.add_parser(
        "tfidf", help="Provides TF-IDF scores for <document id> <term>"
    )
    tfidf_parser.add_argument("doc_id", help="Document id", type=int)
    tfidf_parser.add_argument("term", help="Term")

    idf_parser = subparsers.add_parser(
        "idf", help="Provides the inverse document frequency"
    )
    idf_parser.add_argument("term", help="IDF term")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()
    indexer = InvertedIndex(CACHE_DIR_PATH)

    with open("data/movies.json") as infile:
        movie_data = json.load(infile)

    match args.command:
        case "search":
            query = args.query
            print(f"Searching for: {query}")

            # Load index
            try:
                indexer.load()
            except FileNotFoundError:
                print("Couldn't find cache, perhaps you forgot to build it.")

            search_results = inverted_search(query, indexer)
            for index, search_result in enumerate(search_results):
                print(f"{index + 1}. {search_result}")

        case "build":
            indexer.build(movie_data)
            indexer.save()

        case "tf":
            # Load index
            try:
                indexer.load()
            except FileNotFoundError:
                print("Couldn't find cache, perhaps you forgot to build it.")
            doc_id = args.doc_id
            term = args.term
            print("Term Frequency:", indexer.get_tf(doc_id, term))

        case "idf":
            # Load index
            try:
                indexer.load()
            except FileNotFoundError:
                print("Couldn't find cache, perhaps you forgot to build it.")

            term = args.term
            idf = indexer.get_idf(term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")

        case "tfidf":
            # Load index
            try:
                indexer.load()
            except FileNotFoundError:
                print("Couldn't find cache, perhaps you forgot to build it.")
            doc_id = args.doc_id
            term = args.term
            tf_idf = indexer.get_tf_idf(doc_id, term)
            print(
                f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}"
            )

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
