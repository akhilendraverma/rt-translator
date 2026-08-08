"""
Gateway — the orchestration layer clients talk to.

Proposal mapping
----------------
"audio is streamed from the client via WebRTC, segmented into 500ms chunks,
transcribed by the ASR service, translated by the NMT service, and synthesised
by the TTS service before being streamed back."

This gateway implements that fan-out. WebRTC signalling itself is browser-side;
here the browser sends 500ms audio chunks over a WebSocket (WSS in production),
which is the transport the proposal names for the audio stream.

Endpoints
---------
WS   /ws/translate?source=en&target=es   streaming: audio chunk in, {text, translation, audio} out
POST /translate                           one-shot REST for the same pipeline (used by tests)
GET  /health                              aggregates downstream service health
"""
import base64
import os
import logging

from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("gateway")

ASR_URL = os.getenv("ASR_URL", "http://asr:8001")
NMT_URL = os.getenv("NMT_URL", "http://nmt:8002")
TTS_URL = os.getenv("TTS_URL", "http://tts:8003")


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await client.aclose()


app = FastAPI(title="RT-Translator Gateway", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# Shared async client — connection pooling keeps per-chunk latency down.
# Generous timeout: the FIRST request to each service downloads/loads its model
# (OPUS-MT, Whisper), which can take a minute or more on a slow connection. After
# that it's cached and fast. connect stays short; read is long for cold starts.
client = httpx.AsyncClient(timeout=httpx.Timeout(connect=10.0, read=300.0, write=30.0, pool=10.0))


class TranslateRequest(BaseModel):
    audio_b64: str
    source: str = "auto"
    target: str = "es"


async def run_pipeline(audio_b64: str, source: str, target: str) -> dict:
    """ASR -> NMT -> TTS. Returns transcript, translation, and synthesised audio."""
    asr = (await client.post(
        f"{ASR_URL}/transcribe",
        json={"audio_b64": audio_b64, "language": source},
    )).json()

    text = asr.get("text", "")
    detected = asr.get("language", source)
    if not text.strip():
        return {"text": "", "translation": "", "audio_b64": "", "source": detected}

    src = detected if source == "auto" else source
    nmt = (await client.post(
        f"{NMT_URL}/translate",
        json={"text": text, "source": src, "target": target},
    )).json()
    translation = nmt.get("translation", "")

    tts = (await client.post(
        f"{TTS_URL}/synthesize",
        json={"text": translation, "language": target},
    )).json()

    return {
        "text": text,
        "translation": translation,
        "audio_b64": tts.get("audio_b64", ""),
        "sample_rate": tts.get("sample_rate", 0),
        "source": src,
        "target": target,
    }


@app.get("/health")
async def health():
    out = {"gateway": "ok"}
    for name, url in (("asr", ASR_URL), ("nmt", NMT_URL), ("tts", TTS_URL)):
        try:
            r = await client.get(f"{url}/health", timeout=3.0)
            out[name] = r.json().get("status", "unknown")
        except Exception as e:
            out[name] = f"unreachable ({e.__class__.__name__})"
    return out


@app.post("/translate")
async def translate(req: TranslateRequest):
    """One-shot REST pipeline — same path the WebSocket uses, easier to test."""
    return await run_pipeline(req.audio_b64, req.source, req.target)


@app.websocket("/ws/translate")
async def ws_translate(ws: WebSocket):
    """
    Streaming endpoint. Client sends base64 audio chunks (~500ms each) as text
    frames; gateway replies with a JSON result per chunk. This is where WebRTC
    audio lands after the browser buffers it into chunks.
    """
    await ws.accept()
    params = ws.query_params
    source = params.get("source", "auto")
    target = params.get("target", "es")
    log.info("WS open source=%s target=%s", source, target)
    try:
        while True:
            chunk_b64 = await ws.receive_text()
            try:
                result = await run_pipeline(chunk_b64, source, target)
            except Exception as e:
                # A single slow/failed chunk (e.g. cold model load timing out)
                # must not kill the session. Report it and keep listening.
                log.warning("pipeline error on a chunk: %s", e)
                result = {"text": "", "translation": "",
                          "audio_b64": "", "error": str(e)[:200]}
            await ws.send_json(result)
    except WebSocketDisconnect:
        log.info("WS closed")
