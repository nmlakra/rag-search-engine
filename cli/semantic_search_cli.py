#!/usr/bin/env python3
import argparse
from lib.semantic_search import verify_model, SemanticSearch
from config import SEMANTIC_SEARCH_MODEL
import json


def embed_text(text):
    semantic_search = SemanticSearch(SEMANTIC_SEARCH_MODEL)
    text_embedding = semantic_search.generate_embeddings(text)
    return text_embedding[0]


def verify_embeddings():
    semantic_search = SemanticSearch(SEMANTIC_SEARCH_MODEL)
    with open("data/movies.json", "r") as f:
        documents = json.load(f)
    print(f"Number of docs:   {len(documents)}")
    embeddings = semantic_search.load_or_build_embeddings(documents["movies"])
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )


def embed_query_text(query):
    semantic_search = SemanticSearch(SEMANTIC_SEARCH_MODEL)
    query_embedding = semantic_search.generate_embeddings(query)[0]
    print(f"Query: {query}")
    print(f"First 3 dimensions: {query_embedding[:3]}")
    print(f"Shape: {query_embedding.shape}")
    return query_embedding


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Verify the semantic search model is working
    subparsers.add_parser("verify", help="Verify the semantic search model is working")

    # Verify that embeddings are generated and cached correctly
    subparsers.add_parser(
        "verify_embeddings",
        help="Verify that embeddings are generated and cached correctly",
    )

    # Generate embeddings for the provided text
    embed_text_parser = subparsers.add_parser(
        "embed_text",
        help="Generate embeddings for the provided text",
    )
    embed_text_parser.add_argument(
        "text", type=str, help="Text to generate embeddings for"
    )
    embed_query_parser = subparsers.add_parser(
        "embed_query",
        help="Generate embeddings for the provided query text",
    )
    embed_query_parser.add_argument(
        "query", type=str, help="Query text to generate embeddings for"
    )

    # Search for relevant documents based on the provided query
    search_parser = subparsers.add_parser(
        "search", help="Search for relevant documents based on the provided query"
    )
    search_parser.add_argument(
        "query", type=str, help="Query text to search for relevant documents"
    )
    search_parser.add_argument(
        "--limit", type=int, default=5, help="Number of search results to return"
    )

    # Chuck the positional args for text and chunck based on the chunksize
    chunk_parser = subparsers.add_parser(
        "chunk",
        help="Chunk the provided text into smaller pieces based on the specified chunk size",
    )
    chunk_parser.add_argument("text", type=str, help="Text to be chunked")
    chunk_parser.add_argument(
        "--chunk-size", type=int, default=200, help="Size of each chunk"
    )

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model(SEMANTIC_SEARCH_MODEL)
        case "embed_text":
            text = args.text
            if not text:
                print("Error: No text provided for embedding.")
                return
            embedding = embed_text(text)
            print(f"Text: {text}")
            print(f"First 3 dimensions: {embedding[:3]}")
            print(f"Dimensions: {embedding.shape[0]}")
        case "verify_embeddings":
            verify_embeddings()

        case "embed_query":
            query = args.query
            embed_query_text(query)

        case "search":
            query = args.query
            limit = args.limit
            semantic_search = SemanticSearch(SEMANTIC_SEARCH_MODEL)
            with open("data/movies.json", "r") as f:
                documents = json.load(f)
            semantic_search.load_or_build_embeddings(documents["movies"])
            results = semantic_search.search(query, limit)
            for index, result in enumerate(results):
                print(
                    f"{index + 1}. {result.get('title', 'N/A')} ({result['score']:.4f})\n",
                    result.get("description", "N/A") + "\n",
                )

        case "chunk":
            text = args.text
            chunk_size = args.chunk_size
            if not text:
                print("Error: No text provided for chunking.")
                return
            words = text.split(" ")
            idx = 1
            print(f"Chunking {len(text)} characters")
            for idx, i in enumerate(range(0, len(words), chunk_size), start=1):
                start_pos = i
                end_pos = i + chunk_size
                chunk = " ".join(words[start_pos:end_pos])
                print(f"{idx}. {chunk}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
