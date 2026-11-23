from typing import Literal

from pydantic import BaseModel

Regulation = Literal["AI_ACT", "GDPR"]
ChunkType = Literal["article", "recital", "annex"]

_REGULATION_NAMES = {"AI_ACT": "AI Act", "GDPR": "GDPR"}


class Chunk(BaseModel):
    id: str
    text: str
    regulation: Regulation
    language: Literal["de", "en"]
    type: ChunkType
    number: str
    paragraph: str | None = None
    title: str | None = None
    source_url: str
    celex: str
    version_date: str

    @property
    def key(self) -> tuple[str, str, str, str | None]:
        """Language-independent identity: the same provision in DE and EN shares a key."""
        return (self.regulation, self.type, self.number, self.paragraph)


class Citation(BaseModel):
    regulation: Regulation
    type: ChunkType
    number: str
    paragraph: str | None = None
    title: str | None = None
    source_url: str
    language: str

    @property
    def label(self) -> str:
        name = _REGULATION_NAMES[self.regulation]
        if self.type == "article":
            paragraph = f"({self.paragraph})" if self.paragraph else ""
            return f"{name} Art. {self.number}{paragraph}"
        if self.type == "recital":
            return f"{name} Recital {self.number}"
        # Annex: number is the annex numeral, paragraph is the point.
        point = f" point {self.paragraph}" if self.paragraph else ""
        return f"{name} Annex {self.number}{point}"


class Answer(BaseModel):
    answer: str
    language: str
    citations: list[Citation]
    refused: bool
    latency_ms: int
    input_tokens: int
    output_tokens: int
