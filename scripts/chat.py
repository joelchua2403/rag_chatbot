
from app.rag import answer


def main():
    print("arXiv Version-Aware RAG Chatbot")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            question = input("Question> ").strip()

            if question.lower() in {"quit", "exit"}:
                print("Goodbye!")
                break

            if not question:
                continue

            result = answer(question)

            print("\n" + result["answer"])

            if result.get("sources"):
                print("\nSources:")

                for source in result["sources"]:
                    print(
                        f"  [{source['label']}] "
                        f"arXiv:{source['arxiv_id']}"
                        f"v{source['version']} "
                        f"page {source['page']}"
                    )

            print()

        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        except ValueError as exc:
            print(f"\nInvalid request: {exc}\n")

        except Exception as exc:
            print(f"\nError: {exc}\n")


if __name__ == "__main__":
    main()
