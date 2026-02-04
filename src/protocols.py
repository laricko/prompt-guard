from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import BaseModel

type PhraseSchema = dict[str, list[str]]


class GuardResult[T](BaseModel):
    score: float
    evidence: T


class Guard(Protocol):
    async def check(self, prompt: str) -> GuardResult:
        ...
