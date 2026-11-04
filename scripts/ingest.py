import sys
from app.ingestion import ingest
if __name__ == "__main__":
    ingest(sys.argv[1:] or None)
