from pathlib import Path

import yaml
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from protocols import GuardResult, PhraseSchema


class TfIdfMatch(BaseModel):
    category: str
    phrase: str
    score: float


class TfIdfGuard:
    def __init__(self, phrase_db: str, *, top_k: int = 5) -> None:
        data: PhraseSchema = yaml.safe_load(Path(phrase_db).read_text(encoding="utf-8"))

        self._top_k = top_k
        self._meta: list[tuple[str, str]] = []
        phrases: list[str] = []

        for category, items in data.items():
            for phrase in items:
                phrases.append(phrase)
                self._meta.append((category, phrase))

        self._vectorizer = TfidfVectorizer(ngram_range=(1, 3))
        self._phrase_matrix = self._vectorizer.fit_transform(phrases)

    async def check(self, prompt: str) -> GuardResult[list[TfIdfMatch]]:
        vec = self._vectorizer.transform([prompt])
        sims = cosine_similarity(vec, self._phrase_matrix).ravel()

        matches = self._build_matches(sims)
        score = max(sims, default=0.0)

        return GuardResult(score=score, evidence=matches)

    def _build_matches(self, sims) -> list[TfIdfMatch]:
        indexed_scores = list(enumerate(sims))

        indexed_scores.sort(key=lambda x: (-float(x[1]), x[0]))

        matches: list[TfIdfMatch] = []
        for idx, sim in indexed_scores[: self._top_k]:
            score = float(sim)
            if score <= 0.0:
                continue

            category, phrase = self._meta[idx]
            matches.append(
                TfIdfMatch(
                    category=category,
                    phrase=phrase,
                    score=score,
                )
            )

        return matches
