from app.rag import answer
if __name__ == "__main__":
    print("Type 'quit' to exit")
    while True:
        question = input("Question> ").strip()
        if question.lower() in {"quit", "exit"}:
            break
        if question:
            result = answer(question)
            print("\n" + result["answer"])
            for source in result["sources"]:
                print(f"  [{source['label']}] arXiv:{source['arxiv_id']}v{source['version']} page {source['page']}")
