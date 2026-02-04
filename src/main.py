from fastapi import FastAPI

from tf_idf_guard import TfIdfGuard

app = FastAPI()

@app.post("/check")
async def check(prompt: str):
    tf_idf_guard = TfIdfGuard(phrase_db="phrases.yml")
    return await tf_idf_guard.check(prompt=prompt)
