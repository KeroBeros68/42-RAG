import json
from pathlib import Path
from typing import Any, Optional

import bm25s
import pickle

import chromadb
from sentence_transformers import SentenceTransformer

from src.models.models import MinimalSource, RagDataset, StudentSearchResults


class Retriever:
    DATA_DIR: str = "src/data/processed"
    BM25_INDEX_PATH: str = "src/data/processed/bm25_index"
    CHUNKS_PATH: str = "src/data/processed/chunks.pkl"
    K_COEFFICIENT: int = 5

    SEARCH_RESULT_DIR: str = "src/data/output/search_results/"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    COLLECTION_NAME: str = "Corpus"

    _chroma_retriever: Optional[chromadb.PersistentClient] = None
    _collection: chromadb.Collection | None = None
    _chroma_embedding: SentenceTransformer | None = None
    _bm25_retriever: bm25s.BM25 | None = None
    _chunks: list[MinimalSource] | None = None

    @classmethod
    def _get_chroma_retriever(cls) -> chromadb.PersistentClient:
        if cls._chroma_retriever is None:
            cls._chroma_retriever = chromadb.PersistentClient(
                path=cls.DATA_DIR
            )
        return cls._chroma_retriever

    @classmethod
    def _get_collection(cls) -> chromadb.Collection:
        if cls._collection is None:
            cls._collection = cls._get_chroma_retriever().get_collection(
                cls.COLLECTION_NAME
            )
        return cls._collection

    @classmethod
    def _get_chroma_embedding(cls) -> SentenceTransformer:
        if cls._chroma_embedding is None:
            cls._chroma_embedding = SentenceTransformer(cls.EMBEDDING_MODEL)
        return cls._chroma_embedding

    @classmethod
    def _get_bm25_retriever(cls) -> bm25s.BM25:
        if cls._bm25_retriever is None:
            cls._bm25_retriever = bm25s.BM25().load(
                cls.BM25_INDEX_PATH, load_corpus=True
            )
        return cls._bm25_retriever

    @classmethod
    def _get_chunks(cls) -> list[MinimalSource]:
        if cls._chunks is None:
            with open(cls.CHUNKS_PATH, "rb") as f:
                cls._chunks = pickle.load(f)
        return cls._chunks

    @classmethod
    def search_mode(
        cls, query: str, k: int = 5, chroma: bool = False
    ) -> list[MinimalSource]:

        search_res = cls._search_bm25(query, cls._get_chunks(), k)
        if chroma:
            search_res.extend(cls._search_chromadb(query, k))
        search_res = list(set(search_res))
        search_res.sort(key=lambda x: x.score, reverse=True)
        return search_res

    @classmethod
    def _search_bm25(
        cls,
        query: str,
        chunks: list[MinimalSource],
        k: int = 5,
    ) -> list[MinimalSource]:

        bm25_res: list[MinimalSource] = []

        query_tokens = bm25s.tokenize(
            [query],
            stopwords="en_plus",
            show_progress=True,
        )

        res = cls._get_bm25_retriever().retrieve(
            query_tokens, k=k * cls.K_COEFFICIENT
        )

        results, scores = res
        for i in range(k * cls.K_COEFFICIENT):
            doc_index = results[0, i]
            source: MinimalSource = chunks[doc_index]
            bm25_res.append(
                source.model_copy(update={"score": float(scores[0, i])})
            )

        return cls._harmonize_score(bm25_res)

    @classmethod
    def _search_chromadb(cls, query: str, k: int = 5) -> list[MinimalSource]:
        chroma_res: list[MinimalSource] = []

        embeddings = (
            cls._get_chroma_embedding()
            .encode(query, show_progress_bar=False)
            .tolist()
        )

        res = cls._get_collection().query(
            query_embeddings=embeddings, n_results=k * cls.K_COEFFICIENT
        )

        ids = res["ids"][0]
        raw_metadatas = res["metadatas"]
        raw_distances = res["distances"]
        assert raw_metadatas is not None and raw_distances is not None
        metadatas = raw_metadatas[0]
        distances = raw_distances[0]

        for i in range(len(ids)):
            meta = MinimalSource(
                file_path=str(metadatas[i]["file_path"]),
                first_character_index=int(
                    metadatas[i]["first_character_index"]
                ),
                last_character_index=int(metadatas[i]["last_character_index"]),
            )
            meta.score = 1 / (1 + distances[i])
            chroma_res.append(meta)

        return cls._harmonize_score(chroma_res)

    @staticmethod
    def _harmonize_score(res: list[MinimalSource]) -> list[MinimalSource]:
        if not res:
            return res

        for r in res:
            r.score = 1 / (r.score + 60)

        return res

    @staticmethod
    def print_res(
        sources: list[MinimalSource], query: str, k: int = 5
    ) -> None:
        print(f"Résultats pour : '{query}'")

        for i, source in enumerate(sources):
            print(f"[{i+1}] Score: {source.score:.4f}")
            print(f"    Fichier: {source.file_path}")
            print(
                f"    Position: {source.first_character_index}"
                f" -> {source.last_character_index}"
            )

    @staticmethod
    def read_dataset(path: str) -> dict[str, Any]:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                data_dict = json.load(f)
            return RagDataset.model_validate(data_dict).model_dump()
        except Exception as e:  # A CHANGER
            print(str(e))
            raise

    @classmethod
    def save_search(cls, search_result: str, dataset_path: str) -> None:
        output_dir = Path(cls.SEARCH_RESULT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)

        file_name = Path(dataset_path).name
        output_path = output_dir / file_name
        with open(output_path, "w") as f:
            f.write(search_result)

    @classmethod
    def process_multiple_query(
        cls, dataset: dict[str, Any], k: int, chroma: bool
    ) -> StudentSearchResults:
        res_data: dict[str, Any] = {"k": k, "search_results": []}

        for data in dataset["rag_questions"]:
            search_res: list[MinimalSource] = cls.search_mode(
                data["question"], k, chroma
            )

            data["retrieved_sources"] = [
                source.model_dump() for source in search_res[:k]
            ]

            res_data["search_results"].append(data)

        try:
            return StudentSearchResults.model_validate(res_data)
        except Exception as e:  # A CHANGER
            print(str(e))
            raise
