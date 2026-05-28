from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from tqdm import tqdm

from src.models.models import MinimalSource

RAW_DIR: str = "data/raw"


def load_files() -> list[Document]:
    raw_path = Path(RAW_DIR)
    md_files = list(raw_path.rglob("*.md"))
    py_files = list(raw_path.rglob("*.py"))
    all_files = md_files + py_files

    docs = []

    for file_path in tqdm(all_files, desc="Loading documents"):
        try:
            content = Path(file_path).read_text(encoding="utf-8")
            docs.append(
                Document(
                    page_content=content, metadata={"source": str(file_path)}
                )
            )
        except Exception as e:
            print(f"Error on file {file_path}: {e}")

    return docs


def chunk_file(
    docs_list: list[Document], max_chunk_file: int
) -> tuple[list[Document], list[MinimalSource]]:
    lang_map = {".py": Language.PYTHON}

    chunks: list[Document] = []
    chunks_min_src: list[MinimalSource] = []

    for doc in tqdm(docs_list, desc="Chunking documents"):
        extension = Path(doc.metadata["source"]).suffix

        language = lang_map.get(extension, Language.MARKDOWN)

        splitter = RecursiveCharacterTextSplitter.from_language(
            language=language,
            chunk_size=max_chunk_file,
            chunk_overlap=150,
            add_start_index=True,
        )

        split_docs = splitter.split_documents([doc])

        for d in tqdm(
            split_docs, desc=Path(doc.metadata["source"]).name, leave=False
        ):
            start = int(d.metadata["start_index"])

            metadata = MinimalSource.model_validate(
                {
                    "file_path": doc.metadata["source"],
                    "first_character_index": start,
                    "last_character_index": start + len(d.page_content),
                }
            )
            chunks.append(
                Document(
                    d.page_content,
                    metadata=metadata.model_dump(),
                )
            )
            chunks_min_src.append(metadata)
    return chunks, chunks_min_src
