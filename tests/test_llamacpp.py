"""llama.cpp client checks against a local fake server."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from bbg.llamacpp import Client, InvalidAnswer, parse_answer, run_split


def body(content, reasoning=None):
    msg = {"role": "assistant", "content": content}
    if reasoning is not None:
        msg["reasoning_content"] = reasoning
    return {"choices": [{"message": msg}]}


def test_parse_answer_accepts_only_exact_labels():
    assert parse_answer(body("bullish")) == 1
    assert parse_answer(body(" neutral\n")) == 2
    for bad in ("Bullish", "bullish.", "very bullish", ""):
        with pytest.raises(InvalidAnswer):
            parse_answer(body(bad))
    with pytest.raises(InvalidAnswer):
        parse_answer(body("bearish", reasoning="let me think"))


@pytest.fixture
def server():
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(payload)
            text = payload["messages"][0]["content"]
            answer = "bullish" if "jump" in text else "bearish" if "plunge" in text else "neutral"
            data = json.dumps(body(answer)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}/v1/chat/completions", seen
    httpd.shutdown()


def test_run_split_classifies_and_resumes(server, tmp_path):
    url, seen = server
    client = Client(url, "test-model", api_key="secret")
    ids, texts = ["a", "b", "c"], ["shares jump", "stocks plunge", "fed meeting"]
    cache = tmp_path / "val.jsonl"

    preds, lat = run_split(client, ids[:2], texts[:2], cache)
    assert preds == [1, 0] and len(seen) == 2
    preds, lat = run_split(client, ids, texts, cache)           # resumes: only "c" is requested
    assert preds == [1, 0, 2] and len(seen) == 3 and len(lat) == 3

    p = seen[0]
    assert p["temperature"] == 0 and p["chat_template_kwargs"] == {"enable_thinking": False}
    assert p["grammar"] == 'root ::= "bearish" | "bullish" | "neutral"'


def test_cache_from_another_model_is_refused(server, tmp_path):
    url, _ = server
    cache = tmp_path / "val.jsonl"
    run_split(Client(url, "model-a"), ["a"], ["shares jump"], cache)
    with pytest.raises(RuntimeError):
        run_split(Client(url, "model-b"), ["a"], ["shares jump"], cache)
