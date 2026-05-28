from fire import Fire

from controller import Controller

PROG_NAME: str = "RAG against the machine"
PROG_DESCRIPTION: str = (
    "RAG against the machine — hybrid code retrieval pipeline. "
    "Indexes a codebase with BM25 and semantic embeddings, "
    "then retrieves relevant source passages for natural language queries."
)

PROG_HELP: str = (
    "Commands:\n"
    "  index          Build the retrieval index from raw sources\n"
    "  search         Retrieve top-k passages for a query\n"
    "  search_dataset Run retrieval over a full question dataset\n"
    "\n"
    "Examples:\n"
    "  python -m src index --max_chunk_size=1500 --chroma=True\n"
    "  python -m src search 'how does the cache work' --k=10\n"
    "  python -m src search_dataset --path=code --k=5\n"
    "\n"
    "Run 'python -m src <command> --help' for command-specific options.\n"
    "Dependencies must be installed: make install"
)


Fire(Controller())
