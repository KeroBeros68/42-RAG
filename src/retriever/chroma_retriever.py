from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from pydantic import Field


class ChromaRetriever(BaseRetriever):
    vectorstore: Chroma = Field(description="Chroma vectorstore")
    k: int = Field(default=5, description="Number of results")
    embeddings: HuggingFaceEmbeddings = Field(description="Embedding model")

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_documents(
        cls,
        documents: list[Document],
        k: int = k,
        persist_directory: str = "data/processed/db",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> "ChromaRetriever":
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        vectorstore = Chroma.from_documents(
            documents, embeddings, persist_directory=persist_directory
        )
        return cls(vectorstore=vectorstore, k=k, embeddings=embeddings)

    @classmethod
    def from_index(
        cls,
        persist_directory: str,
        k: int = k,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> "ChromaRetriever":
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
        )
        return cls(vectorstore=vectorstore, k=k, embeddings=embeddings)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        results = self.vectorstore.similarity_search_with_score(
            query, k=self.k
        )
        find_docs = []
        for doc, score in results:
            doc_improved = Document(
                page_content=doc.page_content,
                metadata={
                    **doc.metadata,
                    "chroma_score": round(float(score), 4),
                },
            )
            find_docs.append(doc_improved)
        return find_docs
