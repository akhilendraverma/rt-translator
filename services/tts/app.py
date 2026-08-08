"""
TTS microservice — Text-to-Speech.

Proposal mapping
----------------
Layer : TTS (Coqui XTTS-v2 for natural, low-latency synthesis)
Here  : eSpeak NG called directly (offline, CPU, headless-safe) as the default so
        the stack runs reliably inside a container with no desktop audio system.
        Set TTS_BACKEND=coqui to load XTTS-v2 on a GPU box.

Why espeak and not pyttsx3: pyttsx3 drives a desktop speech service (sapi/nsss/
espeak via a driver layer) and often fails in a headless Docker container. Calling
espeak directly to write a WAV file avoids that entirely.

Contract
--------
POST /synthesize  body: {"text": str, "language": "es"}
                  -> {"audio_b64": "<base64 wav>", "sample_rate": int}
GET  /health      -> {"status": "ok", "backend": ...}
"""
import base64
import io
import os
import logging
import subprocess
import tempfile
import wave

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("tts")

TTS_BACKEND = os.getenv("TTS_BACKEND", "espeak")   # espeak | pyttsx3 | coqui

app = FastAPI(title="TTS Service", version="0.2.0")
_coqui = None

# espeak uses its own language codes; most ISO-639-1 codes work directly.
# A few need mapping.
ESPEAK_LANG = {
    "en": "en", "es": "es", "fr": "fr", "de": "de", "it": "it", "pt": "pt",
    "nl": "nl", "ru": "ru", "pl": "pl", "tr": "tr", "el": "el",
    "zh": "zh", "ja": "ja", "ko": "ko", "hi": "hi", "ar": "ar",
}


class SynthesizeRequest(BaseModel):
    text: str
    language: str = "en"


class SynthesizeResponse(BaseModel):
    audio_b64: str
    sample_rate: int


def _synth_espeak(text: str, language: str) -> tuple[bytes, int]:
    """Call espeak-ng / espeak to write a WAV file. Headless-safe."""
    lang = ESPEAK_LANG.get(language, "en")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        path = f.name
    try:
        # -v voice/language, -w write to WAV file. Try espeak-ng then espeak.
        last_err = None
        ran = False
        for binary in ("espeak-ng", "espeak"):
            try:
                proc = subprocess.run(
                    [binary, "-v", lang, "-w", path, text],
                    check=True, capture_output=True, text=True,
                )
                ran = True
                break
            except FileNotFoundError:
                continue
            except subprocess.CalledProcessError as e:
                last_err = e.stderr or str(e)
                continue
        if not ran:
            raise HTTPException(
                status_code=500,
                detail=f"espeak synthesis failed: {last_err or 'espeak not installed'}",
            )
        # Confirm we actually got a non-empty WAV before returning it.
        if not os.path.exists(path) or os.path.getsize(path) < 44:  # 44 = WAV header size
            raise HTTPException(status_code=500, detail="espeak produced empty audio")
        with wave.open(path, "rb") as w:
            sr = w.getframerate()
        with open(path, "rb") as fh:
            data = fh.read()
        return data, sr
    finally:
        if os.path.exists(path):
            os.unlink(path)


def _synth_pyttsx3(text: str) -> tuple[bytes, int]:
    import pyttsx3
    engine = pyttsx3.init()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        path = f.name
    engine.save_to_file(text, path)
    engine.runAndWait()
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
    with open(path, "rb") as fh:
        data = fh.read()
    os.unlink(path)
    return data, sr


def _synth_coqui(text: str, language: str) -> tuple[bytes, int]:
    global _coqui
    if _coqui is None:
        from TTS.api import TTS
        log.info("Loading Coqui XTTS-v2")
        _coqui = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
    buf = io.BytesIO()
    _coqui.tts_to_file(text=text, language=language, file_path=buf)
    return buf.getvalue(), 24000


@app.get("/health")
def health():
    return {"status": "ok", "backend": TTS_BACKEND}


@app.post("/synthesize", response_model=SynthesizeResponse)
def synthesize(req: SynthesizeRequest):
    if not req.text.strip():
        # Nothing to speak — return a valid empty response, don't error.
        return SynthesizeResponse(audio_b64="", sample_rate=0)
    try:
        if TTS_BACKEND == "coqui":
            audio, sr = _synth_coqui(req.text, req.language)
        elif TTS_BACKEND == "pyttsx3":
            audio, sr = _synth_pyttsx3(req.text)
        else:
            audio, sr = _synth_espeak(req.text, req.language)
    except HTTPException:
        raise
    except Exception as e:
        log.warning("synthesis failed: %s", e)
        # Don't crash the pipeline; return empty audio so the gateway still
        # returns the transcript and translation text.
        return SynthesizeResponse(audio_b64="", sample_rate=0)

    return SynthesizeResponse(
        audio_b64=base64.b64encode(audio).decode("ascii"),
        sample_rate=sr,
    )
