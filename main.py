import argparse
from agent import load_and_index_documents, query_agent

def main():
    parser = argparse.ArgumentParser(description="RAG Agent with Qdrant and Ollama")
    subparsers = parser.add_subparsers(dest="command")

    index_parser = subparsers.add_parser("index", help="Load documents and index into Qdrant")
    query_parser = subparsers.add_parser("query", help="Query the RAG agent")
    query_parser.add_argument("question", type=str, help="The question to ask the agent")

    args = parser.parse_args()

    if args.command == "index":
        print("Indexing documents...")
        load_and_index_documents()
        print("Indexing completed.")
    elif args.command == "query":
        print(f"Question: {args.question}")
        answer, sources = query_agent(args.question)
        print("\nAnswer:\n", answer)
        print("\nSources:\n", sources)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
