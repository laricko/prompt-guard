Prompt guard library for composing multiple safety checks.

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
