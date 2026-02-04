from pathlib import Path

from llama_index.core import Document, VectorStoreIndex
from llama_index.embeddings.ollama import OllamaEmbedding

from protocols import GuardEvidence, GuardResult


class RagGuard:
    def __init__(
        self,
        sentences: str,
        *,
        model_name: str = "mxbai-embed-large",
        top_k: int = 5,
    ) -> None:
        self._top_k = top_k
        self._embed_model = OllamaEmbedding(model_name=model_name)

        raw_lines = Path(sentences).read_text(encoding="utf-8").splitlines()
        lines = [line.strip() for line in raw_lines if line.strip()]
        docs = [Document(text=line) for line in lines]

        self._index = VectorStoreIndex.from_documents(
            docs,
            embed_model=self._embed_model,
        )
        self._retriever = self._index.as_retriever(similarity_top_k=self._top_k)

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
