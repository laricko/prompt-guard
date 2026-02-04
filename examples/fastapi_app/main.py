from contextlib import asynccontextmanager

from fastapi import FastAPI

from prompt_guard import GuardPipeline, PromptGuardConfig


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = PromptGuardConfig()
    app.state.guard = GuardPipeline(config)
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/check")
async def check_prompt(prompt: str):
    guard: GuardPipeline = app.state.guard
    return await guard.check(prompt)
