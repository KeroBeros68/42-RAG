import json
from pathlib import Path
import re


from src.models.models import MinimalSource


class Ingest:
    repo_path: str = "src/data/raw"
    mime_type: list[str] = [
        ".py",
        ".md",
        ".txt",
        ".c",
        ".h"
    ]
    separator: dict[str, list[str]] = {
        ".md": [r"\n#{1,6} ", r"\n\n", r"\n"],
        ".py": [
            r"\n(?=class\s+)",
            r"\n(?=(?:@[\w\.]+\n\s*)*def\s+)",
            r"\n(?=import\s+|from\s+)",
            r"\n\n",
            r"\n",
        ],
        ".c": [
            r"\n(?=[a-zA-Z_][a-zA-Z0-9_]*\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\([^)]*\)\s*\{)",  # noqa E402
            r"\n(?=(struct|enum|union)\s+[a-zA-Z_])",
            r"\n\n",
            r";\n",
            r"\n",
        ],
        ".default": [r"\n\n", r"\n"],
    }

    separator[".h"] = [
        r"\n(?=/\*\*)",
        r"\n(?=#define)",
        *separator[".c"],
    ]

    @classmethod
    def ingest_repository(cls, max_chunk_size) -> list[MinimalSource]:
        all_chunks: list[MinimalSource] = []
        for path in Path(cls.repo_path).rglob("*"):
            if path.suffix in cls.mime_type:
                content = path.read_text(encoding="utf-8")
                chunks = cls.recursive_split(
                    content,
                    0,
                    cls.separator.get(path.suffix, cls.separator[".default"]),
                    max_chunk_size,
                    str(path),
                )
                all_chunks.extend(chunks)
        return all_chunks

    @staticmethod
    def recursive_split(
        text: str,
        start_offset: int,
        separators: list[str],
        max_chunk_size: int,
        path: str,
    ) -> list[MinimalSource]:
        if len(text) <= max_chunk_size:
            return [
                MinimalSource(
                    file_path=path,
                    first_character_index=start_offset,
                    last_character_index=start_offset + len(text),
                )
            ]
        if not separators:
            chunks = []
            for i in range(0, len(text), max_chunk_size):
                chunk_text = text[i: i + max_chunk_size]
                chunks.append(
                    MinimalSource(
                        file_path=path,
                        first_character_index=start_offset + i,
                        last_character_index=start_offset
                        + i
                        + len(chunk_text),
                    )
                )
            return chunks

        current_sep = separators[0]
        remaining_seps = separators[1:]

        parts = re.split(f"(?={current_sep})", text)

        final_chunks = []
        current_buffer = ""
        current_buffer_start = 0
        for i, part in enumerate(parts):
            if len(current_buffer) + len(part) > max_chunk_size:
                if current_buffer:
                    final_chunks.extend(
                        Ingest.recursive_split(
                            current_buffer,
                            start_offset + current_buffer_start,
                            remaining_seps,
                            max_chunk_size,
                            path,
                        )
                    )

                current_buffer = part
                current_buffer_start = sum(len(p) for p in parts[:i])
            else:
                current_buffer += part

        if current_buffer:
            final_chunks.extend(
                Ingest.recursive_split(
                    current_buffer,
                    start_offset + current_buffer_start,
                    remaining_seps,
                    max_chunk_size,
                    path,
                )
            )

        return final_chunks

    @staticmethod
    def save_chunks_to_json(chunks: list[MinimalSource]):
        data_to_save = [chunk.model_dump() for chunk in chunks]
        file_path = "src/data/processed/chunks.json"

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, indent=4, ensure_ascii=False)
