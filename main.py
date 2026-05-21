from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from tqdm import tqdm

from src.retriever.bm25s_retriever import BM25SRetriever

RAW_DIR: str = "data/raw"

md_loader = DirectoryLoader(
    path=RAW_DIR,
    glob="**/*.md",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"},
    recursive=True,
)

py_loader = DirectoryLoader(
    path=RAW_DIR,
    glob="**/*.py",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"},
    recursive=True,
)

docs_md = md_loader.load()
docs_py = py_loader.load()
docs = docs_md + docs_py

lang_map = {".py": Language.PYTHON}

chunks = []
for doc in tqdm(docs, desc="Loading and Chunking"):
    extension = Path(doc.metadata["source"]).suffix

    language = lang_map.get(extension, Language.MARKDOWN)

    splitter = RecursiveCharacterTextSplitter.from_language(
        language=language,
        chunk_size=2000,
        chunk_overlap=150,
        add_start_index=True,
    )

    split_docs = splitter.split_documents([doc])

    for d in tqdm(
        split_docs, desc=Path(doc.metadata["source"]).name, leave=False
    ):
        start = int(d.metadata["start_index"])
        chunks.append(
            Document(
                d.page_content,
                metadata={
                    "source": doc.metadata["source"],
                    "first_character_index": start,
                    "last_character_index": start + len(d.page_content),
                },
            )
        )

bm25_retriever = BM25SRetriever.from_documents(
    chunks,
    k=4,
    langue="english",
    avec_stemming=True,
    index_path="data/processed/index_bm25s",
)

résultats = bm25_retriever.invoke("What command can be used to evaluate the accuracy of a quantized model using lm_eval with vLLM?")
for doc in résultats:
    print(doc.metadata["source"])
    print(doc.metadata["first_character_index"])
    print(doc.metadata["last_character_index"])
    print()
