from pathlib import Path

from src.models.models import MinimalSource


class Ingest:
    repo_path: str = "src/data/raw"
    mime_type: list[str] = [".py", ".md"]
    separator: dict[str, list[str]] = {
        ".md": [r"\n#{1,6} ", r"\n\n", r"\n"],
        ".py": [r"Class", r"def", r"\n\n", r"\n"],
        ".default": [r"\n\n", r"\n"],
    }

    @classmethod
    def ingest_repository(cls, max_chunk_size) -> list[MinimalSource]:
        all_chunks: list[MinimalSource] = []
        for path in Path(cls.repo_path).rglob("*"):
            if path.suffix in cls.mime_type:
                content = path.read_text(encoding="utf-8")
                chunks = cls.smart_split(content, str(path), max_chunk_size)
                all_chunks.extend(chunks)
        return all_chunks

    @staticmethod
    def smart_split(
        content: str, path: str, max_chunk_size: int
    ) -> list[MinimalSource]:
        content_len: int = len(content)
        if content_len < max_chunk_size:
            return [
                MinimalSource(
                    file_path=path,
                    first_character_index=0,
                    last_character_index=content_len,
                )
            ]
        else:
            return []
