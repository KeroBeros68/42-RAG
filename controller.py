from typing import Any

from langchain_huggingface import HuggingFaceEmbeddings
import json

from src.models.models import MinimalSource, RagDataset, StudentSearchResults
from src.file_chunk.indexer import get_bm25_retriever, get_vectoriel_retriever
from src.file_chunk.load_and_chunk import chunk_file, load_files

EMBEDDINGS_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_K: int = 5
DEFAULT_CHUNK_SIZE: int = 2000
DEFAULT_HYBRID_MODE: bool = False

HYBRID_PARAM: list[float] = [0.4, 0.6]  # BM25=40%, vectoriel=60%


class Controller:
    def __init__(self) -> None:
        pass

    def index(
        self,
        k: int = DEFAULT_K,
        max_chunk_size: int = DEFAULT_CHUNK_SIZE,
        hybrid: bool = DEFAULT_HYBRID_MODE,
    ) -> None:

        self.chunks, self.chunks_min_src = chunk_file(
            load_files(), max_chunk_size
        )
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)
        if hybrid:
            self.vectoriel = get_vectoriel_retriever(
                self.chunks, self.embeddings, k, max_chunk_size
            )
        self.bm25_retriever = get_bm25_retriever(
            self.chunks, k, max_chunk_size
        )

    def search(
        self,
        querry: str,
        k: int = DEFAULT_K,
        max_chunk_size: int = DEFAULT_CHUNK_SIZE,
        hybrid: bool = DEFAULT_HYBRID_MODE,
    ) -> None:
        from langchain_classic.retrievers import EnsembleRetriever

        self.index(k, max_chunk_size, hybrid)

        self.retriever = (
            EnsembleRetriever(
                retrievers=[self.bm25_retriever, self.vectoriel],
                weights=HYBRID_PARAM,
            )
            if hybrid
            else self.bm25_retriever
        )

        docs = self.retriever.invoke(querry)[:k]

        files = set(
            [
                MinimalSource(
                    file_path=doc.metadata.get("file_path", ""),
                    first_character_index=doc.metadata.get(
                        "first_character_index", ""
                    ),
                    last_character_index=doc.metadata.get(
                        "last_character_index", ""
                    ),
                )
                for doc in docs
            ]
        )

        for f in files:
            print(f)

    def search_dataset(
        self,
        dataset_path: str,
        k: int = DEFAULT_K,
        max_chunk_size: int = DEFAULT_CHUNK_SIZE,
        hybrid: bool = DEFAULT_HYBRID_MODE,
    ) -> None:
        def read_dataset(path: str) -> dict[str, Any]:
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    data_dict = json.load(f)
                return RagDataset.model_validate(data_dict).model_dump()
            except Exception as e:  # A CHANGER
                print(str(e))
                raise

        def process_multiple_query(
            dataset: dict[str, Any]
        ):
            for data in dataset["rag_questions"]:
                self.search(data.get("question"), k, max_chunk_size, hybrid)

        process_multiple_query(read_dataset(dataset_path))

    def answer(
        self,
        k: int = DEFAULT_K,
        max_chunk_size: int = DEFAULT_CHUNK_SIZE,
        hybrid: bool = DEFAULT_HYBRID_MODE,
    ) -> None:
        pass

    def answer_dataset(
        self,
        k: int = DEFAULT_K,
        max_chunk_size: int = DEFAULT_CHUNK_SIZE,
        hybrid: bool = DEFAULT_HYBRID_MODE,
    ) -> None:
        pass

    def evaluate(self) -> None:
        pass
