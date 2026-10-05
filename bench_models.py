"""Day 5, Part D: compare two models on the same prompts."""

import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

BASE = "https://api.groq.com/openai/v1"
API_KEY = os.getenv("GROQ_API_KEY")

MODELS = [
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b"
]

PROMPTS = [
    "Reply with exactly: OK",
    "In two sentences, what is an AI agent?",
    "A course costs Rs. 18,000 with a 15% scholarship. What is payable? Show the steps."
]

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}


def run(model, prompt):
    start = time.time()

    response = requests.post(
        f"{BASE}/chat/completions",
        headers=HEADERS,
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0
        },
        timeout=600
    )

    data = response.json()
    elapsed = time.time() - start

    text = data["choices"][0]["message"]["content"].strip()
    tokens = data.get("usage", {}).get("completion_tokens", 0)

    rate = tokens / elapsed if elapsed else 0

    return elapsed, tokens, rate, text


if __name__ == "__main__":
    for model in MODELS:
        print("=" * 72)
        print("MODEL:", model)

        for prompt in PROMPTS:
            elapsed, tokens, rate, text = run(model, prompt)

            print(f"\n prompt: {prompt[:50]}")
            print(
                f" {elapsed:5.1f} s | {tokens:4d} tokens | "
                f"{rate:5.1f} tok/s"
            )
            print(f" answer: {text[:160]}")
            print()