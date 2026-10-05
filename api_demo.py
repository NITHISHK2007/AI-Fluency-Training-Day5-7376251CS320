"""Day 5, Part C: call the Groq REST API directly."""

import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

BASE = "https://api.groq.com/openai/v1"
API_KEY = os.getenv("GROQ_API_KEY")
MODEL = os.getenv("MODEL", "openai/gpt-oss-120b")

PROMPT = "In three sentences, explain what an AI agent is."

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}


def generate_once(model=MODEL, prompt=PROMPT):
    """POST /chat/completions: one request, one answer."""
    start = time.time()

    response = requests.post(
        f"{BASE}/chat/completions",
        headers=HEADERS,
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0
        },
        timeout=300
    )

    data = response.json()
    elapsed = time.time() - start

    text = data["choices"][0]["message"]["content"]

    print(f"\n[generate] {elapsed:.1f} seconds")
    print(text.strip())


def chat_streaming(model=MODEL, prompt=PROMPT):
    """Streaming request: measure time to first token."""
    start = time.time()
    first_token_at = None
    pieces = []

    response = requests.post(
        f"{BASE}/chat/completions",
        headers=HEADERS,
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "stream": True
        },
        stream=True,
        timeout=300
    )

    for line in response.iter_lines():
        if not line:
            continue

        line = line.decode("utf-8")

        if line.startswith("data: "):
            line = line[6:]

        if line == "[DONE]":
            break

        import json
        chunk = json.loads(line)

        piece = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")

        if piece and first_token_at is None:
            first_token_at = time.time() - start

        pieces.append(piece)

    total = time.time() - start
    text = "".join(pieces)

    print(
        f"\n[chat, streaming] TTFT {first_token_at:.2f} s | "
        f"total {total:.1f} s"
    )
    print(text.strip())


if __name__ == "__main__":
    generate_once()
    chat_streaming()