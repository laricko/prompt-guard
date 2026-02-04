# Prompt Guard

Composable prompt-safety guards as a lightweight Python library. It provides:
- **TF‑IDF guard** for fast lexical similarity against known phrases.
- **RAG guard** for embedding-based retrieval against a sentence corpus.
- **LLM judge** for a final model-based classification.

## Quick start

```python
from prompt_guard import GuardPipeline, PromptGuardConfig

cfg = PromptGuardConfig(
    phrases_path="phrases.txt",
    sentences_path="sentences.txt",
    embed_model_name="mxbai-embed-large",
    judge_model_name="qwen2.5:3b-instruct",
)
guard = GuardPipeline(cfg)
result = await guard.check("Some prompt")
```

## Notes

- TF‑IDF and RAG build in‑memory indexes at first use.
- RAG and LLM judge use local Ollama models by default; ensure Ollama is running and the models are pulled.

## Text file formats

- `phrases.txt`: one phrase per line (blank lines are ignored).
- `sentences.txt`: one sentence per line (blank lines are ignored).
