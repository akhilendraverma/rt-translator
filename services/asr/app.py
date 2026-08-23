"""
ASR microservice — Automatic Speech Recognition.

Proposal mapping
----------------
Layer : ASR (OpenAI Whisper, fine-tuned)
Here  : `faster-whisper` (CTranslate2 build of Whisper) so it runs on CPU.
        Swap WHISPER_MODEL to "large-v3" on a GPU box to match the proposal exactly.

Contract
--------
POST /transcribe   body: {"audio_b64": "<base64 wav/pcm>", "language": "auto"}
                   -> {"text": str, "language": str, "confidence": float}
GET  /health       -> {"status": "ok", "model": ...}
"""
import base64
import io
import os
import logging

from fastapi import FastAPI
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("asr")

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "tiny")          # tiny|base|small|large-v3
COMPUTE_TYPE = os.getenv("ASR_COMPUTE_TYPE", "int8")        # int8 = CPU-friendly

app = FastAPI(title="ASR Service", version="0.1.0")
_model = None


def get_model():
    """Lazy-load the model so the container starts fast and tests can mock it."""
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        log.info("Loading Whisper model=%s compute=%s", WHISPER_MODEL, COMPUTE_TYPE)
        _model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type=COMPUTE_TYPE)
    return _model


class TranscribeRequest(BaseModel):
    audio_b64: str
    language: str = "auto"          # "auto" lets Whisper detect


class TranscribeResponse(BaseModel):
    text: str
    language: str
    confidence: float


@app.get("/health")
def health():
    return {"status": "ok", "model": WHISPER_MODEL, "compute": COMPUTE_TYPE}


@app.post("/transcribe", response_model=TranscribeResponse)
def transcribe(req: TranscribeRequest):
    audio_bytes = base64.b64decode(req.audio_b64)
    lang = None if req.language in ("auto", "") else req.language

    model = get_model()
    try:
        segments, info = model.transcribe(
            io.BytesIO(audio_bytes),
            language=lang,
            beam_size=1,                # greedy decoding — fastest
            best_of=1,                  # no sampling alternatives — fastest
            temperature=0,              # deterministic, skips fallback decoding passes
            condition_on_previous_text=False,  # don't reprocess prior context each clip
            vad_filter=True,            # drop silence inside the clip
        )
        text = " ".join(s.text.strip() for s in segments).strip()
        conf = float(getattr(info, "language_probability", 0.0) or 0.0)
        detected = getattr(info, "language", lang or "unknown")
    except Exception as e:
        # A streamed chunk can be an undecodable fragment (no container header),
        # silence, or truncated audio. Don't crash the pipeline — just return an
        # empty transcript so the gateway skips this chunk and waits for the next.
        log.warning("transcribe skipped a chunk: %s", e)
        return TranscribeResponse(text="", language=lang or "unknown", confidence=0.0)

    return TranscribeResponse(
        text=text,
        language=detected,
        confidence=round(conf, 4),
    )
