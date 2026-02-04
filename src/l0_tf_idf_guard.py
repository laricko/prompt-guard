from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from protocols import GuardEvidence, GuardResult


class TfIdfGuard:
    def __init__(self, phrase_db: str, *, top_k: int = 5) -> None:
        raw_lines = Path(phrase_db).read_text(encoding="utf-8").splitlines()
        self._phrases = [line.strip() for line in raw_lines if line.strip()]
        self._top_k = top_k

        self._vectorizer = TfidfVectorizer(ngram_range=(1, 3))
        self._phrase_matrix = self._vectorizer.fit_transform(self._phrases)

    async def check(self, prompt: str) -> GuardResult:
        vec = self._vectorizer.transform([prompt])
        sims = cosine_similarity(vec, self._phrase_matrix).ravel()

        matches = self._build_matches(sims)
        score = max(sims, default=0.0)

        return GuardResult(score=score, evidence=matches)

    def _build_matches(self, sims) -> list[GuardEvidence]:
        indexed_scores = list(enumerate(sims))

        indexed_scores.sort(key=lambda x: (-float(x[1]), x[0]))

        matches: list[GuardEvidence] = []
        for idx, sim in indexed_scores[: self._top_k]:
            score = float(sim)
            if score <= 0.0:
                continue

            phrase = self._phrases[idx]
            matches.append(
                GuardEvidence(kind="tfidf", score=score, detail=phrase)
            )

        return matches
