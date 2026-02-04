from typing import Literal, Protocol

from pydantic import BaseModel

type PhraseSchema = dict[str, list[str]]


class GuardEvidence(BaseModel):
    kind: Literal["tfidf", "rag", "judge"]
    score: float
    detail: str


class GuardResult(BaseModel):
    score: float
    evidence: list[GuardEvidence]


class Guard(Protocol):
    async def check(self, prompt: str) -> GuardResult:
        ...
