from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from prompt_guard.protocols import GuardEvidence, GuardResult


class TfIdfGuard:
    def __init__(
        self,
        phrases: list[str],
        vectorizer: TfidfVectorizer,
        phrase_matrix,
        *,
        top_k: int = 5,
    ) -> None:
        self._phrases = phrases
        self._top_k = top_k

        self._vectorizer = vectorizer
        self._phrase_matrix = phrase_matrix

    async def check(self, prompt: str) -> GuardResult:
        vec = self._vectorizer.transform([prompt])
        sims = cosine_similarity(vec, self._phrase_matrix).ravel()

        matches = self._build_matches(sims)
        score = max(sims, default=0.0)

        return GuardResult(score=score, evidence=matches, kind="tfidf")

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
                GuardEvidence(score=score, detail=phrase)
            )

        return matches
