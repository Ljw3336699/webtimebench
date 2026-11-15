"""DeepSeek client for LLM-baseline comparison (document dating).

Adapted from the working call pattern in the expire-dict pipeline
(`FilterLLM`, tianchi proxy). OpenAI-compatible chat/completions:

  POST http://10.11.175.3/tianchi/chat/completions
  headers: Content-Type, appid, Authorization: Bearer <auth>, Host: tianchi-proxy.baidu-int.com
  body:    {model, messages, temperature, top_p, max_tokens, stream}

Credentials are read from environment variables (TIANCHI_APPID / TIANCHI_AUTH)
or from a data-only JSON file ({"appid": ..., "auth": ...}); they are never
printed or written to logs.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import requests

DEFAULT_URL = "http://10.11.175.3/tianchi/chat/completions"
DEFAULT_MODEL = "deepseek-v4.1-flash"
DEFAULT_CREDENTIALS = Path(__file__).resolve().parents[2] / "private" / "tianchi.json"

DATE_PROMPT = (
    "Infer the time when the content was published. If the content mentions related "
    "time, you need to determine whether this time can be used directly. If you find "
    "that no further external knowledge is needed, you can summarize the information "
    "and directly provide the answers in <answer> and </answer> without detailed "
    "explanations. The returned format should be 'YYYY-MM-DD'. Content:"
)


def load_credentials(credentials_file=None):
    """Return (appid, auth) from env or a data-only JSON file."""
    appid = os.environ.get("TIANCHI_APPID")
    auth = os.environ.get("TIANCHI_AUTH")
    if appid and auth:
        return appid, auth
    path = Path(credentials_file) if credentials_file else DEFAULT_CREDENTIALS
    if path.is_file():
        data = json.loads(path.read_text())
        if not isinstance(data, dict) or not data.get("appid") or not data.get("auth"):
            raise ValueError("tianchi credentials file must contain nonempty appid and auth")
        return data["appid"], data["auth"]
    raise ValueError("DeepSeek credentials missing; set TIANCHI_APPID/TIANCHI_AUTH or write private/tianchi.json")


class DeepSeekClient:
    def __init__(self, url=DEFAULT_URL, model=DEFAULT_MODEL, appid=None, auth=None,
                 timeout=(10, 300)):
        self.url = url
        self.model = model
        if not appid or not auth:
            appid, auth = load_credentials()
        self.appid = appid
        self.auth = auth
        self.timeout = timeout

    def chat(self, messages, temperature=0.1, top_p=0.7, max_tokens=4096):
        payload = {"model": self.model, "messages": messages,
                   "temperature": temperature, "top_p": top_p,
                   "max_tokens": max_tokens, "stream": False,
                   "functions": None, "function_call": "none"}
        headers = {"Content-Type": "application/json", "appid": self.appid,
                   "Authorization": "Bearer " + self.auth,
                   "Host": "tianchi-proxy.baidu-int.com"}
        resp = requests.post(self.url, headers=headers, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        msg = resp.json()["choices"][0]["message"]
        content = str(msg.get("content", "") or "").strip()
        reasoning = str(msg.get("reasoning_content", "") or "").strip()
        return content, reasoning


def extract_answer(text):
    m = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL)
    return m.group(1).strip() if m else None


def predict_publication_date(client, body, temperature=0.1, max_tokens=4096):
    """Zero-shot DeepSeek baseline: predict a document's publication date."""
    content, reasoning = client.chat(
        [{"role": "user", "content": DATE_PROMPT + body[:6000]}],
        temperature=temperature, max_tokens=max_tokens)
    return extract_answer(content), content, reasoning


def main():
    import argparse
    ap = argparse.ArgumentParser(description="DeepSeek dating baseline smoke test")
    ap.add_argument("--body", default="NASA announced the Artemis II crew on 2024-04-03.")
    args = ap.parse_args()
    client = DeepSeekClient()
    date, content, reasoning = predict_publication_date(client, args.body)
    print("answer:", date)
    print("content:", content[:200].replace("\n", " "))
    print("reasoning_len:", len(reasoning))


if __name__ == "__main__":
    main()
