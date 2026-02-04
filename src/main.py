from fastapi import FastAPI

from rag_guard import RagGuard
from tf_idf_guard import TfIdfGuard

app = FastAPI()

@app.post("/check")
async def check(prompt: str):
    tf_idf_guard = TfIdfGuard(phrase_db="phrases.txt")
    rag_guard = RagGuard("sentences.txt")
    return await rag_guard.check(prompt=prompt)
    return await tf_idf_guard.check(prompt=prompt)
