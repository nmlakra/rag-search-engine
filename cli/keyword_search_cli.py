import argparse
import json
from typing import Dict, List
from lib.text_preprocessor import text_preprocessor
from lib.inverted_index import InvertedIndex
from config import CACHE_DIR_PATH


def search(movie_data: Dict, query: str, search_limit: int = 5) -> List:
    search_results = []
    for movie in movie_data["movies"]:
        movie_title = movie["title"]
        title_tokens = text_preprocessor(movie_title)
        query_tokens = text_preprocessor(query)

        matched = any(q in t for t in title_tokens for q in query_tokens)

        if matched:
            search_results.append(movie_title)
        if len(search_results) >= search_limit:
            break
    return search_results


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

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
