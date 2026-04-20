from pathlib import Path
import re


import bm25s  # type: ignore
import chromadb
import pickle
from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


from src.models.models import MinimalSource


class Indexer:

    EXCLUDE_DIRS = {".git", ".venv", "__pycache__", "build", "dist", "test"}
    DEFAULT_SAVE_DIR: str = "src/data/processed"

    # chunk const
    MAX_CHUNK_SIZE: int = 2000

    OVERLAP_COEFFICIENT: float = 0.3

    # chromadb const
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    COLLECTION_NAME: str = "Corpus"

    BATCH_SIZE: int = 256

    @classmethod
    def load_and_chunk(cls, repo_path: str, chunk_size: int = MAX_CHUNK_SIZE):
        all_chunks = []
        files = []
        for ext in ["*.py", "*.md", "*.c", "*.cpp", "*.h", "*.toml"]:
            for p in Path(repo_path).rglob(ext):
                if not any(
                    excluded in p.parts for excluded in cls.EXCLUDE_DIRS
                ):
                    files.append(p)

        for file_path in tqdm(files, desc="Loading and Chunking..."):
            try:
                with open(
                    file_path, "r", encoding="utf-8", errors="ignore"
                ) as f:
                    content = f.read()
                if not content.strip():
                    continue
                relative_path = str(file_path.relative_to(repo_path))
                chunks = cls.chunk_file(relative_path, content, chunk_size)
                all_chunks.extend(chunks)
            except Exception as e:  # A CHANGERRRRRRRRRRRRRR
                print(f"{e}")

        print(f"\n✅ {len(files)} files → {len(all_chunks)} chunks")
        return all_chunks

    @classmethod
    def chunk_file(
        cls, file_path: str, content: str, chunk_size: int = MAX_CHUNK_SIZE
    ):
        extension = Path(file_path).suffix

        overlap = int(chunk_size * cls.OVERLAP_COEFFICIENT)

        lang_map = {
            ".py": Language.PYTHON,
            ".c": Language.C,
            ".cpp": Language.CPP,
        }
        language = lang_map.get(extension, Language.MARKDOWN)

        splitter = RecursiveCharacterTextSplitter.from_language(
            language=language,
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            add_start_index=True,  # ✅ LangChain track les positions
        )

        doc = Document(page_content=content, metadata={"source": file_path})
        split_docs = splitter.split_documents([doc])

        return [
            {
                "file_path": file_path,
                "text": d.page_content,
                "first_character_index": d.metadata["start_index"],
                "last_character_index": d.metadata["start_index"]
                + len(d.page_content),
            }
            for d in split_docs
        ]

    @classmethod
    def build_bm25_index(
        cls, chunks: list[dict], save_dir: str = DEFAULT_SAVE_DIR
    ) -> None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        def clean_text(text):
            text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
            text = re.sub(r"[^\w\s]", " ", text).replace("_", " ")
            return text.lower()

        corpus_cleaned = [clean_text(c["text"]) for c in chunks]
        corpus_tokens = bm25s.tokenize(
            corpus_cleaned,
            stopwords="en_plus",
            show_progress=True,
        )

        retriever = bm25s.BM25(k1=1.2, b=0.85)
        retriever.index(corpus_tokens, show_progress=True)
        retriever.save(f"{save_dir}/bm25_index")

        with open(f"{save_dir}/chunks.pkl", "wb") as f:
            metadata = [
                MinimalSource(
                    file_path=c["file_path"],
                    first_character_index=c["first_character_index"],
                    last_character_index=c["last_character_index"],
                )
                for c in chunks
            ]
            pickle.dump(metadata, f)

        print(f"✅ Index BM25 saved in {save_dir}/")

    @classmethod
    def build_chromadb_index(
        cls, chunks: list[dict], save_dir: str = DEFAULT_SAVE_DIR
    ) -> None:
        retriever = chromadb.PersistentClient(path=save_dir)

        model = SentenceTransformer(cls.EMBEDDING_MODEL)
        try:
            retriever.delete_collection(name=cls.COLLECTION_NAME)
        except Exception:
            pass
        corpus = retriever.create_collection(name=cls.COLLECTION_NAME)

        batch_size = cls.BATCH_SIZE
        for i in tqdm(
            range(0, len(chunks), batch_size), desc="Indexing ChromaDB..."
        ):
            batch = chunks[i: i + batch_size]

            texts = [c["text"].lower() for c in batch]

            embeddings = model.encode(texts, show_progress_bar=False).tolist()

            corpus.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=[
                    {
                        "file_path": c["file_path"],
                        "first_character_index": c["first_character_index"],
                        "last_character_index": c["last_character_index"],
                    }
                    for c in batch
                ],
                ids=[f"id{i + j}" for j in range(len(batch))],
            )
        print(f"✅ ChromaDB index saved in {save_dir}/")
