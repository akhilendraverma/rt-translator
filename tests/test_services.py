"""
Unit + integration tests mirroring the proposal's testing strategy:
  - Unit: each microservice tested independently with mocked models.
  - Integration: gateway pipeline tested with the downstream calls mocked.

Run:  pytest -q   (from the repo root, with each service's deps importable,
                   or just the light ones — heavy models are mocked out).
"""
import base64
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]


def _load(module_path: str, name: str):
    """Import a service's app.py by file path without polluting sys.modules names."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, ROOT / module_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# --------------------------- ASR unit test ---------------------------
def test_asr_transcribe_mocked():
    asr = _load("services/asr/app.py", "asr_app")

    fake_segment = MagicMock(text="hello world")
    fake_info = MagicMock(language="en", language_probability=0.97)
    fake_model = MagicMock()
    fake_model.transcribe.return_value = ([fake_segment], fake_info)

    with patch.object(asr, "get_model", return_value=fake_model):
        client = TestClient(asr.app)
        audio = base64.b64encode(b"RIFF....fake").decode()
        r = client.post("/transcribe", json={"audio_b64": audio, "language": "auto"})
        assert r.status_code == 200
        body = r.json()
        assert body["text"] == "hello world"
        assert body["language"] == "en"
        assert body["confidence"] == pytest.approx(0.97, abs=1e-3)


# --------------------------- NMT unit test ---------------------------
def test_nmt_translate_mocked():
    nmt = _load("services/nmt/app.py", "nmt_app")

    fake_pipe = MagicMock(return_value=[{"translation_text": "hola mundo"}])
    with patch.object(nmt, "_opus_pipeline", return_value=fake_pipe):
        client = TestClient(nmt.app)
        r = client.post("/translate", json={"text": "hello world", "source": "en", "target": "es"})
        assert r.status_code == 200
        assert r.json()["translation"] == "hola mundo"


def test_nmt_same_language_is_passthrough():
    nmt = _load("services/nmt/app.py", "nmt_app2")
    client = TestClient(nmt.app)
    r = client.post("/translate", json={"text": "hi", "source": "en", "target": "en"})
    assert r.json()["translation"] == "hi"


# --------------------------- TTS unit test ---------------------------
def test_tts_synthesize_mocked():
    tts = _load("services/tts/app.py", "tts_app")
    with patch.object(tts, "_synth_pyttsx3", return_value=(b"fake-wav-bytes", 22050)):
        client = TestClient(tts.app)
        r = client.post("/synthesize", json={"text": "hola", "language": "es"})
        assert r.status_code == 200
        body = r.json()
        assert base64.b64decode(body["audio_b64"]) == b"fake-wav-bytes"
        assert body["sample_rate"] == 22050


# ----------------------- Gateway integration -------------------------
def test_gateway_pipeline_mocked():
    gw = _load("gateway/app.py", "gateway_app")

    async def fake_pipeline(audio_b64, source, target):
        return {"text": "hello", "translation": "hola", "audio_b64": "AAA=",
                "source": "en", "target": target}

    with patch.object(gw, "run_pipeline", side_effect=fake_pipeline):
        client = TestClient(gw.app)
        r = client.post("/translate", json={"audio_b64": "x", "source": "auto", "target": "es"})
        assert r.status_code == 200
        body = r.json()
        assert body["translation"] == "hola"
        assert body["target"] == "es"
