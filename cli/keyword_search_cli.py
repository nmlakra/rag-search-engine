import argparse
import json
from typing import Dict, List
from lib.text_preprocessor import text_preprocessor


def search(movie_data: Dict, query: str) -> List:
    search_results = []
    search_counter = 0
    for movie in movie_data["movies"]:
        movie_title = movie["title"]
        if query in text_preprocessor(movie_title):
            search_results.append(movie_title)
            search_counter += 1
        if search_counter >= 5:
            break
    return search_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    with open("data/movies.json") as infile:
        movie_data = json.load(infile)

    match args.command:
        case "search":
            query = args.query
            print(f"Searching for: {query}")
            search_results = search(movie_data, query)
            for index, search_result in enumerate(search_results):
                print(f"{index + 1}. {search_result}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
