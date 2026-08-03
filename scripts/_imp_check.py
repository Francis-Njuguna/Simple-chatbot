import sys

sys.path.insert(0, ".")
print("1 chunker", flush=True)
from backend.app.ingest.chunker import TextChunker  # noqa: E402

print("2 pipeline", flush=True)
from backend.app.ingest.pipeline import IngestionPipeline  # noqa: E402

print("3 retriever", flush=True)
from backend.app.rag.retriever import get_retriever  # noqa: E402

print("4 all imports ok", flush=True)
