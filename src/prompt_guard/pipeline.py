from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from llama_index.core import Document, VectorStoreIndex
from llama_index.embeddings.ollama import OllamaEmbedding
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import PromptGuardConfig
from .guards.judge_guard import LlmJudgeGuard
from .guards.rag_guard import RagGuard
from .guards.tfidf_guard import TfIdfGuard
from .protocols import Guard, GuardEvidence, GuardResult


class GuardPipeline:
    def __init__(self, config: PromptGuardConfig) -> None:
        self._config = config
        self._guards: list[Guard] | None = None
        self._limits: dict[str, float] | None = None

    async def check(self, prompt: str) -> GuardResult:
        guards = self._get_guards()
        evidence: list[GuardEvidence] = []
        score = 0.0

        for guard in guards:
            result = await guard.check(prompt)
            evidence.extend(result.evidence)
            score = max(score, result.score)

        return GuardResult(score=score, evidence=evidence)

    def _get_guards(self) -> list[Guard]:
        if self._guards is None:
            self._validate_paths()
            self._guards = self._build_guards()
        return self._guards

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
        phrases = self._load_lines(self._config.phrases_path, label="phrases_path")
        vectorizer = TfidfVectorizer(ngram_range=self._config.tfidf_ngram_range)
        phrase_matrix = vectorizer.fit_transform(phrases)
        return TfIdfGuard(
            phrases,
            vectorizer,
            phrase_matrix,
            top_k=self._config.tfidf_top_k,
        )

    def _build_rag_guard(self) -> RagGuard:
        lines = self._load_lines(self._config.sentences_path, label="sentences_path")
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

    def _load_lines(self, path: str | None, *, label: str) -> list[str]:
        file_path = self._validate_file(path, label=label)
        return [
            line.strip()
            for line in file_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def _validate_file(self, path: str | None, *, label: str) -> Path:
        if not path:
            raise ValueError(f"{label} must be set.")

        file_path = Path(path)
        if not file_path.is_file():
            raise FileNotFoundError(f"{label} not found: {file_path}")

        return file_path

    def _validate_paths(self) -> None:
        if self._config.enable_tfidf:
            self._validate_file(self._config.phrases_path, label="phrases_path")
        if self._config.enable_rag:
            self._validate_file(self._config.sentences_path, label="sentences_path")
