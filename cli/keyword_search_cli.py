import argparse
import json
from typing import List
from lib.text_preprocessor import text_preprocessor
from lib.inverted_index import InvertedIndex
from config import CACHE_DIR_PATH, BM25_K1, BM25_B


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

    bm25_idf_parser = subparsers.add_parser(
        "bm25idf", help="Get BM25 IDF score for a given term"
    )
    bm25_idf_parser.add_argument(
        "term", type=str, help="Term to get BM25 IDF score for"
    )

    bm25_tf_parser = subparsers.add_parser(
        "bm25tf", help="Provides BM25 term frequencies for <document id> <term>"
    )
    bm25_tf_parser.add_argument("doc_id", help="Document id", type=int)
    bm25_tf_parser.add_argument("term", help="Term")
    bm25_tf_parser.add_argument(
        "k1", type=float, nargs="?", default=BM25_K1, help="Tunable BM25 K1 parameter"
    )
    bm25_tf_parser.add_argument(
        "b", type=float, nargs="?", default=BM25_B, help="Tunable BM25 b parameter"
    )

    bm25_search_parser = subparsers.add_parser(
        "bm25search", help="Search movies using BM25 algorithm"
    )
    bm25_search_parser.add_argument("query", type=str, help="Search query")
    bm25_search_parser.add_argument(
        "--limit", type=int, default=5, help="Number of search results to return"
    )

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

        case "bm25idf":
            # Load index
            try:
                indexer.load()
            except FileNotFoundError:
                print("Couldn't find cache, perhaps you forgot to build it.")

            # doc_id = args.doc_id
            term = args.term
            print(f"Term: {term}")
            bm25idf = indexer.get_bm25_idf(term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")

        case "bm25tf":
            # Load index
            try:
                indexer.load()
            except FileNotFoundError:
                print("Couldn't find cache, perhaps you forgot to build it.")
            doc_id = args.doc_id
            term = args.term
            k1 = args.k1
            b = args.b
            bm25tf = indexer.get_bm25_tf(doc_id, term, k1, b)
            print(
                f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}"
            )

        case "bm25search":
            # Load index
            try:
                indexer.load()
            except FileNotFoundError:
                print("Couldn't find cache, perhaps you forgot to build it.")
            query = args.query
            limit = args.limit
            print(f"Searching for: {query}")

            search_results = indexer.bm25_search(query, limit, BM25_K1, BM25_B)
            for index, search_result in enumerate(search_results):
                doc_id = search_result[0]
                doc_score = search_result[1]
                doc_title = indexer.docmap[doc_id].title
                print(f"{index + 1}. ({doc_id}) {doc_title} - (Score: {doc_score:.2f})")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
