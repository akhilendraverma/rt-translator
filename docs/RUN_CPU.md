# Running on CPU (laptop, no GPU)

This build runs entirely on a laptop with no GPU — Docker Desktop is the only
requirement. The defaults are already CPU-friendly (Whisper `tiny`, OPUS-MT,
offline pyttsx3 TTS), so there are **no code changes to make**.

## Steps

1. Make sure Docker Desktop is installed and running (the whale icon shows
   "Engine running").

2. From this project folder, start everything:

   ```bash
   cp .env.example .env
   docker compose up --build
   ```

   The first run downloads the CUDA-free base images and the small models
   (a few hundred MB, a few minutes). When all four services report healthy,
   the pipeline is up. Leave the terminal open.

3. Open `web/index.html` in Chrome. Leave the gateway address as
   `ws://localhost:8000`, click **Start**, allow microphone access, and speak.
   You'll see the transcript and translation, and hear the translation read back.

4. To stop: press `Ctrl+C` in the terminal. Everything runs locally — no cloud,
   no cost.

## Notes

- Translation quality is lower than the GPU build (`tiny` + OPUS-MT are light
  models). The architecture is identical; on GPU hardware the production models
  (Whisper large-v3, SeamlessM4T-v2, XTTS-v2) load via environment variables —
  see `README` and `docs/ARCHITECTURE.md`.
- To test without a microphone, POST a base64 WAV to `http://localhost:8000/translate`
  or use the interactive API docs at `http://localhost:8000/docs`.
