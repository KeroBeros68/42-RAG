import copy
from logging import Logger

from src.Indexer import Indexer
from src.Retriever import Retriever
from src.models.models import MinimalSource


class Controller:
    RAW_DIR_PATH: str = "src/data/raw/"
    PROCESSED_DIR_PATH: str = "src/data/processed/"
    DATASET_DIR_PATH: str = "src/data/datasets/"
    DEFAULT_UNANSWERED_CODE_DATASET: str = (
        DATASET_DIR_PATH + "UnansweredQuestions/dataset_code_public.json"
    )
    DEFAULT_UNANSWERED_DOCS_DATASET: str = (
        DATASET_DIR_PATH + "UnansweredQuestions/dataset_docs_public.json"
    )
    SEARCH_RESULT_DIR: str = "src/data/output/search_results/"

    def __init__(self, logger: Logger) -> None:
        self.logger: Logger = logger

    def index(self, max_chunk_size: int = 2000, chroma: bool = False) -> None:
        print("max chunk size ", max_chunk_size)
        self.logger.info(f"max chunk size {max_chunk_size}")

        all_chunks = Indexer.load_and_chunk(
            self.RAW_DIR_PATH, max_chunk_size
        )
        Indexer.build_bm25_index(copy.deepcopy(all_chunks))
        if chroma:
            Indexer.build_chromadb_index(all_chunks)
        self.logger.info(
            "Ingestion complete! Indices saved under data/processed/"
        )
        print("Ingestion complete! Indices saved under data/processed/")

    def search(
        self,
        query: str,
        k: int = 5,
        chroma: bool = False,
        show_display: bool = True,
    ) -> list[MinimalSource] | None:
        self.logger.info(f"Search Mode\nQuery: {query}")
        search_res = Retriever.search_mode(query, k, chroma)
        if show_display:
            Retriever.print_res(search_res, query, k)
        else:
            return search_res
        return None

    def search_dataset(
        self,
        path: str = "code",
        k: int = 5,
        chroma: bool = False,
    ) -> None:
        match path:
            case "code":
                path = self.DEFAULT_UNANSWERED_CODE_DATASET
            case "docs":
                path = self.DEFAULT_UNANSWERED_DOCS_DATASET
            case _:
                pass
        self.logger.info(f"Search Dataset Mode\nDataset: {path}")

        dataset = Retriever.read_dataset(path)

        res = Retriever.process_multiple_query(dataset, k, chroma)
        Retriever.save_search(res.model_dump_json(indent=4), path)

    def answer(self) -> None:
        pass

    def answer_dataset(self) -> None:
        pass

    def evaluate(self) -> None:
        pass
