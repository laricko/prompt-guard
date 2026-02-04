from pydantic import BaseModel, Field


BUILTIN_DEFAULT_PHRASES_RESOURCE = "data/phrases.txt"
BUILTIN_DEFAULT_SENTENCES_RESOURCE = "data/sentences.txt"


class PromptGuardConfig(BaseModel):
    phrases_path: str | None = Field(
        default=None,
        examples=["/path/to/phrases.txt"],
        description="Path to phrases file (one phrase per line).",
    )
    sentences_path: str | None = Field(
        default=None,
        examples=["/path/to/sentences.txt"],
        description="Path to sentences file (one sentence per line).",
    )

    tfidf_top_k: int = Field(
        default=5,
        examples=[5],
        description="Number of top TF-IDF matches to keep.",
    )
    tfidf_ngram_range: tuple[int, int] = Field(
        default=(1, 3),
        examples=[(1, 3)],
        description="N-gram range for TF-IDF vectorizer.",
    )
    rag_top_k: int = Field(
        default=5,
        examples=[5],
        description="Number of top RAG matches to retrieve.",
    )

    embed_model_name: str = Field(
        default="mxbai-embed-large",
        examples=["mxbai-embed-large"],
        description="Ollama embedding model name.",
    )
    judge_model_name: str = Field(
        default="qwen2.5:3b-instruct",
        examples=["qwen2.5:3b-instruct"],
        description="Ollama judge model name.",
    )
    judge_temperature: float = Field(
        default=0.0,
        examples=[0.0],
        description="LLM judge temperature.",
    )
    judge_max_tokens: int = Field(
        default=256,
        examples=[256],
        description="LLM judge max tokens.",
    )

    base_url: str | None = Field(
        default=None,
        examples=["http://localhost:11434"],
        description="Optional base URL for Ollama.",
    )

    enable_tfidf: bool = Field(
        default=True,
        examples=[True],
        description="Enable TF-IDF guard.",
    )
    enable_rag: bool = Field(
        default=True,
        examples=[True],
        description="Enable RAG guard.",
    )
    enable_judge: bool = Field(
        default=True,
        examples=[True],
        description="Enable LLM judge guard.",
    )
