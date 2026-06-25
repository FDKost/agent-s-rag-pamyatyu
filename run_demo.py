import sys
from config import Config
from rag_pipeline import get_chain

def main():
    config = Config()
    chain = get_chain(config)
    print("RAG Demo. Type 'exit' to quit.")
    while True:
        try:
            query = input(">> ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if query.lower() in ("exit", "quit"):
            print("Goodbye!")
            break
        result = chain.run(query)
        print(result)

if __name__ == "__main__":
    main()
