from dataclasses import dataclass


@dataclass(slots=True)
class PromptGuardConfig:
    phrases_path: str | None = "phrases.txt"
    sentences_path: str | None = "sentences.txt"

    tfidf_top_k: int = 5
    tfidf_ngram_range: tuple[int, int] = (1, 3)
    rag_top_k: int = 5

    embed_model_name: str = "mxbai-embed-large"
    judge_model_name: str = "qwen2.5:3b-instruct"
    judge_temperature: float = 0.0
    judge_max_tokens: int = 256

    base_url: str | None = None

    tfidf_limit: float = 0.0
    rag_limit: float = 0.0
    judge_limit: float = 0.0

    enable_tfidf: bool = True
    enable_rag: bool = True
    enable_judge: bool = True
