from fastapi import FastAPI

from l1_rag_guard import RagGuard
from l0_tf_idf_guard import TfIdfGuard
from l2_judge_guard import L2JudgeGuard

app = FastAPI()

@app.post("/check/tfidf")
async def check_tfidf(prompt: str):
    tf_idf_guard = TfIdfGuard(phrase_db="phrases.txt")
    return await tf_idf_guard.check(prompt=prompt)


@app.post("/check/rag")
async def check_rag(prompt: str):
    rag_guard = RagGuard("sentences.txt")
    return await rag_guard.check(prompt=prompt)


@app.post("/check/judge-l2")
async def check_judge_l2(prompt: str):
    judge_guard = L2JudgeGuard()
    return await judge_guard.check(prompt=prompt)
