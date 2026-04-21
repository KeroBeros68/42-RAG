import json
from pathlib import Path

import bm25s  # type: ignore
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
    # chromadb const
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    COLLECTION_NAME: str = "Corpus"

    @classmethod
    def search_mode(cls, query, k=5, chroma=False):
        chunks = cls._open_chunks_file()
        search_res = cls._search_bm25(query, chunks, k)
        if chroma:
            search_res.extend(cls._search_chromadb(query, k))
        search_res = list(set(search_res))
        search_res.sort(key=lambda x: x.score, reverse=True)
        return search_res

    @classmethod
    def _open_chunks_file(cls):
        with open(cls.CHUNKS_PATH, "rb") as f:
            metadata = pickle.load(f)
        return metadata

    @classmethod
    def _search_bm25(
        cls,
        query: str,
        chunks,
        k: int = 5,
    ) -> list[MinimalSource]:

        bm25_res: list[MinimalSource] = []
        retriever = bm25s.BM25().load(cls.BM25_INDEX_PATH, load_corpus=True)

        query_tokens = bm25s.tokenize(
            [query],
            stopwords="en_plus",
            show_progress=True,
        )

        res = retriever.retrieve(query_tokens, k=k * cls.K_COEFFICIENT)

        results, scores = res
        for i in range(k * cls.K_COEFFICIENT):
            doc_index = results[0, i]
            source: MinimalSource = chunks[doc_index]
            source.score = float(scores[0, i])
            bm25_res.append(source)

        return cls._harmonize_score(bm25_res)

    @classmethod
    def _search_chromadb(cls, query, k=5):
        chroma_res: list[MinimalSource] = []
        chroma_retriever = chromadb.PersistentClient(path=cls.DATA_DIR)

        collection = chroma_retriever.get_collection(cls.COLLECTION_NAME)

        model = SentenceTransformer(cls.EMBEDDING_MODEL)
        embeddings = model.encode(query, show_progress_bar=False).tolist()

        res = collection.query(
            query_embeddings=embeddings, n_results=k * cls.K_COEFFICIENT
        )

        ids = res["ids"][0]
        metadatas = res["metadatas"][0]
        distances = res["distances"][0]

        for i in range(len(ids)):
            meta: MinimalSource = MinimalSource(**metadatas[i])
            meta.score = 1 / (1 + distances[i])
            chroma_res.append(meta)

        return cls._harmonize_score(chroma_res)

    @staticmethod
    def _harmonize_score(res: list[MinimalSource]):
        if not res:
            return res

        scores = [r.score for r in res]
        min_s, max_s = min(scores), max(scores)
        range_s = max_s - min_s

        for r in res:
            if range_s == 0:
                r.score = 1.0
            else:
                r.score = (r.score - min_s) / range_s

        return res

    @staticmethod
    def print_res(sources: list[MinimalSource], query: str, k: int = 5):
        print(f"Résultats pour : '{query}'")

        for i, source in enumerate(sources):
            print(f"[{i+1}] Score: {source.score:.4f}")
            print(f"    Fichier: {source.file_path}")
            print(
                f"    Position: {source.first_character_index}"
                f" -> {source.last_character_index}"
            )

    @staticmethod
    def read_dataset(path):
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                data_dict = json.load(f)
            return RagDataset.model_validate(data_dict).model_dump()
        except Exception as e:  # A CHANGER
            print(str(e))
            raise

    @classmethod
    def save_search(cls, search_result: str, dataset_path: str):
        output_dir = Path(cls.SEARCH_RESULT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)

        file_name = Path(dataset_path).name
        output_path = output_dir / file_name
        with open(output_path, "w") as f:
            f.write(search_result)

    @classmethod
    def process_multiple_querry(cls, dataset, k, chroma):
        res_data = {"k": k, "search_results": []}

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
