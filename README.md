# AI Real-Time Translator — Microservices Scaffold

A runnable scaffold for the capstone proposal *AI Real-Time Translator*. It implements the proposal's microservices architecture — a gateway that streams audio over WebSocket and fans out to independent **ASR → NMT → TTS** services — with **CPU-friendly model defaults** so the whole stack runs on a laptop. Every heavy GPU model named in the proposal has a documented upgrade switch.

This is a **scaffold**, not a finished capstone: the structure, contracts, Docker wiring, tests, and evaluation harness are here so you can fill in each phase against a working skeleton.

## Architecture

```
                        ┌──────────────────────────────────────────┐
  browser (WebRTC) ──►  │  Gateway  (FastAPI, :8000)                │
   500ms audio chunks   │  • WS /ws/translate  • POST /translate    │
        over WSS        └───────┬───────────┬───────────┬──────────┘
                                │           │           │
                         POST /transcribe   │      POST /synthesize
                                ▼           ▼           ▼
                          ┌─────────┐ ┌──────────┐ ┌─────────┐
                          │  ASR    │ │   NMT    │ │  TTS    │
                          │ :8001   │ │  :8002   │ │ :8003   │
                          │ Whisper │ │ OPUS-MT/ │ │pyttsx3/ │
                          │         │ │ Seamless │ │ Coqui   │
                          └─────────┘ └──────────┘ └─────────┘
```

Each service is its own FastAPI app, its own Docker image, and independently scalable — matching the proposal's "each service is containerised and independently scalable" requirement.

## Quick start (CPU / laptop)

```bash
cp .env.example .env
docker compose up --build
```

Then open `web/index.html` in Chrome, point it at `ws://localhost:8000`, and click **Start**. Allow microphone access; speak; the translated audio plays back.

First run downloads the Whisper `tiny` and one OPUS-MT model (a few hundred MB total). Subsequent runs are cached.

### Run without Docker (dev)

Each service runs standalone:

```bash
# in three terminals, from each service dir, after pip install -r requirements.txt
uvicorn app:app --port 8001   # services/asr
uvicorn app:app --port 8002   # services/nmt
uvicorn app:app --port 8003   # services/tts
# then the gateway, pointed at localhost:
ASR_URL=http://localhost:8001 NMT_URL=http://localhost:8002 TTS_URL=http://localhost:8003 \
  uvicorn app:app --port 8000  # gateway/
```

## Scaling to the full proposal spec

The defaults trade fidelity for "runs on a laptop." Flip these env vars (see `.env.example`) on GPU hardware to match the proposal exactly:

| Layer | Laptop default | Proposal target | Switch |
|---|---|---|---|
| ASR | Whisper `tiny`, int8 | Whisper `large-v3`, fp16 | `WHISPER_MODEL`, `ASR_COMPUTE_TYPE` |
| NMT | OPUS-MT (the proposal's fallback) | SeamlessM4T-v2 | `NMT_BACKEND=seamless` |
| TTS | pyttsx3 (offline) | Coqui XTTS-v2 | `TTS_BACKEND=coqui` |

Horizontal scaling (proposal: ~20 streams per pod):

```bash
docker compose up --scale asr=3 --scale nmt=2
```

## Testing & evaluation

The proposal's metrics each have a harness:

```bash
pytest -q                                   # unit + integration tests (models mocked)
python scripts/benchmark_latency.py --wav sample.wav --n 50   # P50/P95/P99 vs 800ms
python scripts/eval_bleu.py --tsv flores_en_es.tsv --source en --target es   # BLEU vs 38
locust -f scripts/locustfile.py --host http://localhost:8000  # 500 concurrent users
```

## Monitoring

```bash
docker compose --profile monitoring up
```

Prometheus at `:9090`, Grafana at `:3000`. To populate the proposal's WER/BLEU/latency dashboards, add a `/metrics` endpoint to each service (e.g. `prometheus-fastapi-instrumentator`) — noted inline in `monitoring/prometheus.yml`.

## Layout

```
rt-translator/
├── gateway/            # orchestrator: WebSocket + REST, fans out to services
├── services/
│   ├── asr/            # Whisper (faster-whisper on CPU)
│   ├── nmt/            # OPUS-MT / SeamlessM4T
│   └── tts/            # pyttsx3 / Coqui XTTS-v2
├── web/                # WebRTC streaming client
├── scripts/            # latency, BLEU, load-test harnesses
├── tests/              # pytest unit + integration
├── monitoring/         # Prometheus config
├── docs/               # ARCHITECTURE.md — full proposal mapping
├── docker-compose.yml
└── .env.example
```

## What's stubbed vs. real

- **Real and working:** service contracts, gateway orchestration, WebSocket streaming, Docker wiring, CPU inference for all three stages, the test suite, the eval scripts.
- **Left for you (the capstone work):** fine-tuning Whisper with LoRA, swapping in SeamlessM4T/Coqui on GPU, the FLORES-200 dataset slice, Grafana dashboards, WSS/TLS termination, and the ROC/A-B testing analyses. See `docs/ARCHITECTURE.md` for where each hooks in.

See `docs/ARCHITECTURE.md` for the detailed component-by-component mapping to the proposal.
