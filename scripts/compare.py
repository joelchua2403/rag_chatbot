import argparse
from app.comparison import compare
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("arxiv_id")
    parser.add_argument("--old", type=int)
    parser.add_argument("--new", type=int)
    parser.add_argument("--max-changes", type=int, default=30)
    args = parser.parse_args()
    print(compare(args.arxiv_id, args.old, args.new, args.max_changes).model_dump_json(indent=2))
