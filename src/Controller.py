from logging import Logger

from src.Indexer import Indexer
from src.Retriever import Retriever


class Controller:
    def __init__(self, logger) -> None:
        self.logger: Logger = logger

    def index(self, max_chunk_size=2000, chroma=False) -> None:
        print("max chunk size ", max_chunk_size)
        self.logger.info(f"max chunk size {max_chunk_size}")

        all_chunks = Indexer.load_and_chunk("src/data/raw/", max_chunk_size)
        Indexer.build_bm25_index(all_chunks)
        if chroma:
            Indexer.build_chromadb_index(all_chunks)
        self.logger.info(
            "Ingestion complete! Indices saved under data/processed/"
        )
        print("Ingestion complete! Indices saved under data/processed/")

    def search(self, query: str):
        self.logger.info(f"Search Mode\nQuerry: {query}")
        Retriever.search_mode(query)

    def search_dataset(self):
        pass

    def answer(self):
        pass

    def answer_dataset(self):
        pass

    def evaluate(self):
        pass
