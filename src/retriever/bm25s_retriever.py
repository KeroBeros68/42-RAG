from typing import Any, Optional

import bm25s
import Stemmer
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field


class BM25SRetriever(BaseRetriever):
    bm25_index: bm25s.BM25 = Field(description="Index bm25s")
    documents: list[Document] = Field(
        description="Documents LangChain originaux"
    )
    k: int = Field(default=4, description="Nombre de résultats à retourner")
    stemmer: Optional[Any] = Field(
        default=None, description="Stemmer PyStemmer"
    )
    stopwords: str = Field(default="en_plus")

    class Config:
        arbitrary_types_allowed = True

    # ── Constructeurs ─────────────────────────────────────

    @classmethod
    def from_documents(
        cls,
        documents: list[Document],
        k: int = 4,
        langue: str = "english",
        avec_stemming: bool = True,
        index_path: Optional[str] = None,
        stopwords: str = "en_plus",
    ) -> "BM25SRetriever":

        stemmer = Stemmer.Stemmer(langue) if avec_stemming else None
        textes = [doc.page_content for doc in documents]

        corpus_tokenisé = bm25s.tokenize(
            textes,
            stemmer=stemmer,
            stopwords=stopwords,
            show_progress=True,
        )

        index = bm25s.BM25()
        index.index(corpus_tokenisé)

        # Sauvegarder si un chemin est fourni
        if index_path:
            index.save(index_path, corpus=textes)

        return cls(bm25_index=index, documents=documents, k=k, stemmer=stemmer, stopwords=stopwords)

    @classmethod
    def from_index(
        cls,
        index_path: str,
        documents: list[Document],
        k: int = 4,
        langue: str = "english",
    ) -> "BM25SRetriever":
        index = bm25s.BM25.load(index_path, load_corpus=False)
        stemmer = Stemmer.Stemmer(langue)
        return cls(bm25_index=index, documents=documents, k=k, stemmer=stemmer)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:

        query_tokenisée = bm25s.tokenize(
            [query],
            stemmer=self.stemmer,
            stopwords=self.stopwords,
        )

        n = min(self.k, len(self.documents))
        résultats_idx, scores = self.bm25_index.retrieve(query_tokenisée, k=n)

        docs_trouvés = []
        for i in range(résultats_idx.shape[1]):
            idx = int(résultats_idx[0, i])
            score = float(scores[0, i])
            doc = self.documents[idx]

            doc_enrichi = Document(
                page_content=doc.page_content,
                metadata={**doc.metadata, "bm25s_score": round(score, 4)},
            )
            docs_trouvés.append(doc_enrichi)

        return docs_trouvés
