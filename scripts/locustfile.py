"""
Load test the gateway (proposal: 500 concurrent users, measure P95).
Run: locust -f scripts/locustfile.py --host http://localhost:8000
Then open http://localhost:8089 and set users=500.
Set SAMPLE_WAV_B64 env or edit below to a real base64 wav.
"""
import os
import base64

from locust import HttpUser, task, between

SAMPLE = os.getenv("SAMPLE_WAV_B64", base64.b64encode(b"RIFFfake").decode())


class TranslateUser(HttpUser):
    wait_time = between(0.5, 2.0)

    @task
    def translate(self):
        self.client.post("/translate", json={
            "audio_b64": SAMPLE, "source": "auto", "target": "es",
        })
