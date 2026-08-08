"""
NMT microservice — Neural Machine Translation.

Proposal mapping
----------------
Layer   : NMT (Meta SeamlessM4T primary, Helsinki OPUS-MT lightweight fallback)
Here    : OPUS-MT (the proposal's own fallback) because it runs on CPU in seconds.
          On a GPU box, set NMT_BACKEND=seamless to load SeamlessM4T-v2 instead.

Contract
--------
POST /translate  body: {"text": str, "source": "en", "target": "es"}
                 -> {"translation": str, "source": str, "target": str}
GET  /health     -> {"status": "ok", "backend": ...}
"""
import os
import logging
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("nmt")

NMT_BACKEND = os.getenv("NMT_BACKEND", "opus")   # opus | seamless

app = FastAPI(title="NMT Service", version="0.1.0")


class TranslateRequest(BaseModel):
    text: str
    source: str = "en"       # ISO-639-1 codes: en, es, fr, de, ...
    target: str = "es"


class TranslateResponse(BaseModel):
    translation: str
    source: str
    target: str


@lru_cache(maxsize=8)
def _opus_pipeline(source: str, target: str):
    """One OPUS-MT model exists per language pair, so cache per (src,tgt)."""
    from transformers import pipeline
    model_name = f"Helsinki-NLP/opus-mt-{source}-{target}"
    log.info("Loading OPUS-MT %s", model_name)
    try:
        return pipeline("translation", model=model_name, device=-1)  # -1 = CPU
    except Exception as e:  # pragma: no cover
        raise HTTPException(
            status_code=422,
            detail=f"No OPUS-MT model for {source}->{target} ({e}). "
                   f"Try a supported pair or switch NMT_BACKEND=seamless.",
        )


def _translate_opus(text, source, target):
    out = _opus_pipeline(source, target)(text, max_length=512)
    return out[0]["translation_text"]


def _translate_seamless(text, source, target):
    # Heavier; intended for GPU. Imported lazily so CPU installs don't need it.
    from transformers import AutoProcessor, SeamlessM4Tv2ForTextToText
    global _sl_proc, _sl_model
    if "_sl_model" not in globals():
        name = "facebook/seamless-m4t-v2-large"
        log.info("Loading SeamlessM4T %s", name)
        globals()["_sl_proc"] = AutoProcessor.from_pretrained(name)
        globals()["_sl_model"] = SeamlessM4Tv2ForTextToText.from_pretrained(name)
    inputs = _sl_proc(text=text, src_lang=source, return_tensors="pt")
    tokens = _sl_model.generate(**inputs, tgt_lang=target)[0]
    return _sl_proc.decode(tokens, skip_special_tokens=True)


@app.get("/health")
def health():
    return {"status": "ok", "backend": NMT_BACKEND}


@app.post("/translate", response_model=TranslateResponse)
def translate(req: TranslateRequest):
    if not req.text.strip():
        return TranslateResponse(translation="", source=req.source, target=req.target)
    if req.source == req.target:
        return TranslateResponse(translation=req.text, source=req.source, target=req.target)

    if NMT_BACKEND == "seamless":
        translation = _translate_seamless(req.text, req.source, req.target)
    else:
        translation = _translate_opus(req.text, req.source, req.target)

    return TranslateResponse(translation=translation, source=req.source, target=req.target)
