import argparse
import json
from create_agent import create_agent
from knowledge_base_tools import add_to_knowledge_base

def main():
    parser = argparse.ArgumentParser(description="RAG Agent CLI")
    parser.add_argument("--add", action="store_true", help="Add a document to the knowledge base")
    parser.add_argument("--doc", type=str, help="Document text to add")
    parser.add_argument("--meta", type=str, help="Metadata as JSON string")
    parser.add_argument("--query", type=str, help="Query to ask the agent")
    args = parser.parse_args()

    agent = create_agent()

    if args.add and args.doc:
        metadata = json.loads(args.meta) if args.meta else {}
        print(add_to_knowledge_base(document=args.doc, metadata=metadata))

    if args.query:
        response = agent.run(args.query)
        print("Agent response:")
        print(response)

if __name__ == "__main__":
    main()
