# Architecture & Proposal Mapping

This document ties each part of the scaffold back to the *AI Real-Time Translator* proposal, so it's clear what is implemented, what is a lighter stand-in, and where the remaining capstone work plugs in.

## 1. Pipeline overview

The proposal specifies: *"audio is streamed from the client via WebRTC, segmented into 500 ms chunks, transcribed by the ASR service, translated by the NMT service, and synthesised by the TTS service before being streamed back. Each service is containerised (Docker) and independently scalable."*

The scaffold implements exactly this flow:

1. **Client** (`web/index.html`) captures microphone audio with `getUserMedia` (WebRTC) and uses `MediaRecorder` with a 500 ms timeslice to emit chunks.
2. Chunks are base64-encoded and sent over a **WebSocket** to the gateway (`ws://…/ws/translate`). In production this is WSS/TLS 1.3, as the proposal's data-protection section requires.
3. The **gateway** (`gateway/app.py`) runs `run_pipeline()`: it calls ASR, then NMT, then TTS over HTTP, and streams the JSON result back on the same socket.
4. Each stage is a **separate FastAPI service** with its own Dockerfile and port, so they scale independently (`docker compose up --scale asr=3`).

## 2. Component mapping

| Proposal layer | Proposal choice | Scaffold implementation | File | Upgrade switch |
|---|---|---|---|---|
| ASR | Whisper Large-v3, fine-tuned (LoRA) | `faster-whisper` (`tiny`, int8) on CPU | `services/asr/app.py` | `WHISPER_MODEL=large-v3`, `ASR_COMPUTE_TYPE=float16` |
| NMT | SeamlessM4T-v2 primary, OPUS-MT fallback | OPUS-MT (the proposal's own fallback) on CPU; SeamlessM4T path stubbed in | `services/nmt/app.py` | `NMT_BACKEND=seamless` |
| TTS | Coqui XTTS-v2 | pyttsx3 (offline, no download); Coqui path stubbed in | `services/tts/app.py` | `TTS_BACKEND=coqui` |
| Front end | React (web), React Native (mobile), WebRTC | Single-file WebRTC + WebSocket client | `web/index.html` | Port to React when the UI grows |
| Cloud | AWS ECS, auto-scaling | Docker Compose (portable to ECS) | `docker-compose.yml` | Deploy the same images to ECS |
| MLOps | MLflow, Docker, CI/CD, Prometheus | Docker + Prometheus config + tests | `monitoring/`, `tests/` | Add MLflow tracking in training scripts |

**Why the substitutions:** the target hardware is a CPU laptop. The proposal's GPU models (Whisper large-v3, SeamlessM4T, XTTS-v2) need 8–16 GB VRAM to hit the latency targets. Every substitution is a documented one-line switch, and the SeamlessM4T and Coqui code paths are already written (behind the env flags) so upgrading is loading the model, not rewriting the service.

## 3. Service contracts

Stable JSON contracts mean any stage can be swapped without touching the others.

**ASR** — `POST /transcribe`
```
in : {"audio_b64": str, "language": "auto"|iso639-1}
out: {"text": str, "language": str, "confidence": float}
```

**NMT** — `POST /translate`
```
in : {"text": str, "source": iso639-1, "target": iso639-1}
out: {"translation": str, "source": str, "target": str}
```

**TTS** — `POST /synthesize`
```
in : {"text": str, "language": iso639-1}
out: {"audio_b64": str, "sample_rate": int}
```

**Gateway** — `POST /translate` (one-shot) and `WS /ws/translate?source=&target=` (streaming), both returning
```
{"text": str, "translation": str, "audio_b64": str, "sample_rate": int, "source": str, "target": str}
```

## 4. Testing strategy (proposal §4)

| Proposal test type | Where |
|---|---|
| Unit tests per microservice with mock inputs | `tests/test_services.py` — ASR/NMT/TTS each tested with the model mocked |
| Integration test of end-to-end pipeline | `tests/test_services.py::test_gateway_pipeline_mocked` |
| Latency benchmark (< 800 ms, P95 over sessions) | `scripts/benchmark_latency.py` |
| BLEU on FLORES-200 (≥ 38) | `scripts/eval_bleu.py` |
| Load test (500 concurrent, Locust) | `scripts/locustfile.py` |
| ROC / A-B testing | Not yet — hooks: add confidence output from ASR (already returned) to sweep thresholds |

## 5. Data protection (proposal §2)

The scaffold defaults to the proposal's stance — no persistent audio storage; chunks are processed in memory and discarded. To implement the full policy:

- **Transport:** terminate WSS/TLS 1.3 at the gateway (add a reverse proxy such as Caddy or nginx).
- **PII stripping:** insert an NER pass between ASR and NMT in `run_pipeline()`.
- **Opt-in retention:** add a consent flag to the WebSocket query params; only persist when true.

## 6. What remains for the capstone

The scaffold gets you a running end-to-end system on day one. The research contributions the proposal is graded on are deliberately left open:

- Fine-tune Whisper with LoRA adapters on domain audio (ASR service loads the adapter).
- Benchmark SeamlessM4T-v2 vs OPUS-MT for the BLEU ≥ 38 target (flip `NMT_BACKEND`, run `eval_bleu.py`).
- Integrate Coqui XTTS-v2 and measure MOS (flip `TTS_BACKEND`, run the user survey).
- Build the Grafana dashboards from the Prometheus metrics.
- Run the latency optimisation and edge-case fixes of Phase 6.

Each of these plugs into a defined seam, so the engineering scaffolding never blocks the research.
