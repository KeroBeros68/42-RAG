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
        description="Original LangChain documents"
    )
    k: int = Field(default=5, description="Number of result")
    stemmer: Optional[Any] = Field(
        default=None, description="Stemmer PyStemmer"
    )
    stopwords: str = Field(default="en_plus")

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_documents(
        cls,
        documents: list[Document],
        k: int = k,
        langue: str = "english",
        with_stemming: bool = True,
        index_path: Optional[str] = None,
        stopwords: str = "en_plus",
    ) -> "BM25SRetriever":

        stemmer = Stemmer.Stemmer(langue) if with_stemming else None
        texts = [doc.page_content for doc in documents]

        corpus_tokenize = bm25s.tokenize(
            texts,
            stemmer=stemmer,
            stopwords=stopwords,
            show_progress=True,
        )

        index = bm25s.BM25()
        index.index(corpus_tokenize)

        if index_path:
            index.save(index_path, corpus=texts)

        return cls(
            bm25_index=index,
            documents=documents,
            k=k,
            stemmer=stemmer,
            stopwords=stopwords,
        )

    @classmethod
    def from_index(
        cls,
        index_path: str,
        documents: list[Document],
        k: int = k,
        langue: str = "english",
        with_stemming: bool = True,
        stopwords: str = "en_plus",
    ) -> "BM25SRetriever":
        index = bm25s.BM25.load(index_path, load_corpus=False)
        stemmer = Stemmer.Stemmer(langue) if with_stemming else None
        return cls(
            bm25_index=index,
            documents=documents,
            k=k,
            stemmer=stemmer,
            stopwords=stopwords,
        )

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:

        query_tokenize = bm25s.tokenize(
            [query],
            stemmer=self.stemmer,
            stopwords=self.stopwords,
        )

        n = min(self.k, len(self.documents))
        res_idx, scores = self.bm25_index.retrieve(query_tokenize, k=n)

        find_docs = []
        for i in range(res_idx.shape[1]):
            idx = int(res_idx[0, i])
            score = float(scores[0, i])
            doc = self.documents[idx]

            doc_improve = Document(
                page_content=doc.page_content,
                metadata={**doc.metadata, "bm25s_score": round(score, 4)},
            )
            find_docs.append(doc_improve)

        return find_docs
