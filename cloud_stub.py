"""
Cloud parking stub  —  replaces the old bot instance on Render.
===============================================================
The real bot runs LOCALLY. The old cloud copy ("the ghost") kept hijacking
the Telegram webhook and answering the owner with stale/empty data, so this
stub deliberately replaces it: it serves a harmless status page, touches
NOTHING (no Telegram, no trading, no DB), and exists only so the Render
service has something valid to run.

To bring the real bot back to the cloud later: fix the full deploy, then
point the Dockerfile CMD back to  uvicorn main:app .
"""

from fastapi import FastAPI

app = FastAPI()

_MSG = {
    "status": "parked",
    "note": "Cloud instance intentionally suspended - the real bot runs locally. "
            "Restore by pointing Dockerfile CMD back to main:app.",
}


@app.get("/")
async def root():
    return _MSG


@app.get("/ping")
async def ping():
    return _MSG


@app.get("/health")
async def health():
    return _MSG
