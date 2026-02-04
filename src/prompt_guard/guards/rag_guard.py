from llama_index.core.retrievers import BaseRetriever

from ..protocols import GuardEvidence, GuardResult


class RagGuard:
    def __init__(
        self,
        retriever: BaseRetriever,
        *,
        top_k: int = 5,
    ) -> None:
        self._top_k = top_k
        self._retriever = retriever

    async def check(self, prompt: str) -> GuardResult:
        nodes = self._retriever.retrieve(prompt)

        matches = self._build_matches(nodes)
        score = matches[0].score if matches else 0.0

        return GuardResult(score=score, evidence=matches)

    def _build_matches(self, nodes) -> list[GuardEvidence]:
        matches: list[GuardEvidence] = []

        for item in nodes:
            score = float(item.score or 0.0)
            if score <= 0.0:
                continue

            text = item.node.get_content()
            matches.append(GuardEvidence(kind="rag", score=score, detail=text))

        return matches[: self._top_k]
