from collections.abc import Sequence
import uuid

from pydantic import BaseModel, Field


class MinimalSource(BaseModel):
    """Represents the location of a source passage within a file.

    Attributes:
        file_path: Path to the source file.
        first_character_index: Index of the first character of the relevant
        passage.
        last_character_index: Index of the last character of the relevant
        passage.
    """

    file_path: str
    first_character_index: int
    last_character_index: int
    score: float = Field(default=0)

    def __str__(self) -> str:
        return (
            f"path: {self.file_path} start at {self.first_character_index}"
            f" end at {self.last_character_index}"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MinimalSource):
            return False
        return (
            self.file_path == other.file_path
            and self.first_character_index == other.first_character_index
            and self.last_character_index == other.last_character_index
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.file_path,
                self.first_character_index,
                self.last_character_index,
            )
        )


class UnansweredQuestion(BaseModel):
    """Represents a question that has not yet been answered.

    Attributes:
        question_id: Unique identifier for the question, auto-generated as a
        UUID.
        question: The question text.
    """

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """Represents a question that has been answered, along with its supporting
    sources.

    Inherits all fields from UnansweredQuestion and adds the answer and its
    sources.

    Attributes:
        sources: List of source passages that support the answer.
        answer: The answer text.
    """

    sources: list[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """Represents a dataset of RAG (Retrieval-Augmented Generation) questions.

    Questions may be answered or unanswered. Answered questions are validated
    first due to their stricter schema.

    Attributes:
        rag_questions: List of questions, each either answered or unanswered.
    """

    rag_questions: list[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """Represents the retrieval results for a single question.

    Attributes:
        question_id: Unique identifier linking these results to their
        originating question.
        question: The question text that was used to perform the search.
        retrieved_sources: List of source passages retrieved for the question.
    """

    question_id: str
    question: str
    retrieved_sources: list[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Represents the retrieval results for a question together with a
    generated answer.

    Inherits all fields from MinimalSearchResults and adds the generated
    answer.

    Attributes:
        answer: The answer generated from the retrieved sources.
    """

    answer: str


class StudentSearchResults(BaseModel):
    """Represents a collection of retrieval results produced by a student
    retriever.

    Attributes:
        search_results: Sequence of per-question retrieval results.
        k: Number of sources retrieved per question.
    """

    search_results: Sequence[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(StudentSearchResults):
    """Represents a collection of retrieval results together with generated
    answers.

    Narrows the search_results field from the base class to require that every
    entry also carries an answer.

    Attributes:
        search_results: List of per-question retrieval results, each including
        an answer.
    """

    search_results: Sequence[MinimalAnswer]
