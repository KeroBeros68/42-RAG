from logging import Logger

from src.Indexer import Indexer
from src.Retriever import Retriever


class Controller:
    RAW_DIR_PATH: str = "src/data/raw/"
    PROCESSED_DIR_PATH: str = "src/data/processed/"
    DATASET_DIR_PATH: str = "src/data/dataset/"
    DEFAULT_UNANSWERED_CODE_DATASET: str = (
        DATASET_DIR_PATH + "UnansweredQuestions/dataset_code_public.json"
    )
    DEFAULT_UNANSWERED_DOCS_DATASET: str = (
        DATASET_DIR_PATH + "UnansweredQuestions/dataset_docs_public.json"
    )

    def __init__(self, logger) -> None:
        self.logger: Logger = logger

    def index(self, max_chunk_size=2000, chroma=False) -> None:
        print("max chunk size ", max_chunk_size)
        self.logger.info(f"max chunk size {max_chunk_size}")

        all_chunks = Indexer.load_and_chunk(self.RAW_DIR_PATH, max_chunk_size)
        Indexer.build_bm25_index(all_chunks)
        if chroma:
            Indexer.build_chromadb_index(all_chunks)
        self.logger.info(
            "Ingestion complete! Indices saved under data/processed/"
        )
        print("Ingestion complete! Indices saved under data/processed/")

    def search(self, query: str, k: int = 5, chroma: bool = False):
        self.logger.info(f"Search Mode\nQuerry: {query}")
        search_res = Retriever.search_mode(query, k, chroma)
        Retriever.print_res(search_res, query, k)

    def search_dataset(self, dataset: str, k: int = 5, chroma: bool = False):
        match dataset:
            case "code":
                dataset = self.DEFAULT_UNANSWERED_CODE_DATASET
            case "docs":
                dataset = self.DEFAULT_UNANSWERED_DOCS_DATASET
            case _:
                pass
        self.logger.info(f"Search Dataset Mode\nDataset: {dataset}")

    def answer(self):
        pass

    def answer_dataset(self):
        pass

    def evaluate(self):
        pass
