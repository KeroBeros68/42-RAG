from logging import Logger

from src.Ingest import Ingest


class Controller:
    def __init__(self, logger) -> None:
        self.logger: Logger = logger

    def index(self, max_chunk_size=2000) -> None:
        print("max chunk size ", max_chunk_size)
        self.logger.info(f"max chunk size {max_chunk_size}")
        all_chunks = Ingest.ingest_repository(max_chunk_size)
        Ingest.save_chunks_to_json(all_chunks)
        self.logger.info(
            "Ingestion complete! Indices saved under data/processed/"
        )
        self.logger.info(
            f"Total chunks: {len(all_chunks)}"
        )
        print("Ingestion complete! Indices saved under data/processed/")

    def search(self):
        pass

    def search_dataset(self):
        pass

    def answer(self):
        pass

    def answer_dataset(self):
        pass

    def evaluate(self):
        pass
