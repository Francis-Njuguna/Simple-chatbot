import requests

url = "http://localhost:11434/api/generate"

payload = {
    "model": "qwen3:4b",
    "prompt": "Say hello in one sentence.",
    "stream": False,
}

r = requests.post(url, json=payload)

print("Status:", r.status_code)
print(r.text)