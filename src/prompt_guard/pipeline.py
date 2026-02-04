from pathlib import Path

from llama_index.core import Document, VectorStoreIndex
from llama_index.embeddings.ollama import OllamaEmbedding
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import PromptGuardConfig
from .guards.judge_guard import LlmJudgeGuard
from .guards.rag_guard import RagGuard
from .guards.tfidf_guard import TfIdfGuard
from .protocols import Guard


class GuardPipeline:
    def __init__(self, config: PromptGuardConfig) -> None:
        self._config = config
        self._guards = self._build_guards()

    async def check(self, prompt: str):
        results = []

        for guard in self._guards:
            result = await guard.check(prompt)
            results.append(result)

        return results

    def _build_guards(self) -> list[Guard]:
        guards: list[Guard] = []
        if self._config.enable_tfidf:
            guards.append(self._build_tfidf_guard())
        if self._config.enable_rag:
            guards.append(self._build_rag_guard())
        if self._config.enable_judge:
            guards.append(self._build_judge_guard())
        return guards

    def _build_tfidf_guard(self) -> TfIdfGuard:
        phrases = self._load_lines(self._config.phrases_path)
        vectorizer = TfidfVectorizer(ngram_range=self._config.tfidf_ngram_range)
        phrase_matrix = vectorizer.fit_transform(phrases)
        return TfIdfGuard(
            phrases,
            vectorizer,
            phrase_matrix,
            top_k=self._config.tfidf_top_k,
        )

    def _build_rag_guard(self) -> RagGuard:
        lines = self._load_lines(self._config.sentences_path)
        docs = [Document(text=line) for line in lines]
        embed_kwargs = {"model_name": self._config.embed_model_name}
        if self._config.base_url:
            embed_kwargs["base_url"] = self._config.base_url
        embed_model = OllamaEmbedding(**embed_kwargs)
        index = VectorStoreIndex.from_documents(docs, embed_model=embed_model)
        retriever = index.as_retriever(similarity_top_k=self._config.rag_top_k)
        return RagGuard(retriever, top_k=self._config.rag_top_k)

    def _build_judge_guard(self) -> LlmJudgeGuard:
        return LlmJudgeGuard(
            model_name=self._config.judge_model_name,
            temperature=self._config.judge_temperature,
            max_tokens=self._config.judge_max_tokens,
            base_url=self._config.base_url,
        )

    def _load_lines(self, path: str) -> list[str]:
        file_path = Path(path)
        return [
            line.strip()
            for line in file_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
