import os

BM25_INDEX = "data/processed/index_bm25s"

PERSIST_DIR = "data/processed/db"


def get_bm25_retriever(chunks, k: int, max_chunk_size: int):
    from src.retriever.bm25s_retriever import BM25SRetriever

    bm25_index = BM25_INDEX + f"_{max_chunk_size}"
    if os.path.exists(bm25_index):
        return BM25SRetriever.from_index(
            bm25_index,
            chunks,
            k=k,
            langue="english",
            with_stemming=True,
        )
    else:
        return BM25SRetriever.from_documents(
            chunks,
            k=k,
            langue="english",
            with_stemming=True,
            index_path=bm25_index,
        )


def get_vectoriel_retriever(chunks, embeddings, k: int, max_chunk_size: int):
    from langchain_chroma import Chroma

    chroma_index = PERSIST_DIR + f"_{max_chunk_size}"
    if os.path.exists(chroma_index) and os.listdir(chroma_index):
        vectorstore = Chroma(
            persist_directory=chroma_index, embedding_function=embeddings
        )
    else:
        vectorstore = Chroma.from_documents(
            chunks, embeddings, persist_directory=chroma_index
        )

    return vectorstore.as_retriever(search_kwargs={"k": k})
