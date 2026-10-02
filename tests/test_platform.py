import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


class FakePlatform(BaseHTTPRequestHandler):
    state = None

    def log_message(self, *args):
        pass

    def _body(self):
        length = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(length) if length else b""

    def _json(self, status, payload, headers=None):
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _error(self, status, code, headers=None, **extra):
        self._json(status, {"error": {"message": code, "type": "platform_error", "code": code, **extra}}, headers)

    def do_GET(self):
        s = self.state
        s["requests"].append(("GET", self.path, self.headers.get("Authorization")))
        if self.path.startswith("/blob/"):
            self.send_response(200)
            self.send_header("Content-Length", "5")
            self.end_headers()
            self.wfile.write(b"CACT!")
        elif self.path.startswith("/v1/") and self.headers.get("Authorization") != "Bearer needle_ft_test":
            self._error(401, "unauthorized", url="https://cactuscompute.com/dashboard/api-keys")
        elif self.path == "/v1/billing":
            if s["rate_limit_once"]:
                s["rate_limit_once"] = False
                self._error(429, "rate_limited", {"Retry-After": "1"})
            else:
                self._json(200, {"plan": "standard", "limits": {"jobs": 10}, "usage": {"jobs": 1}})
        elif self.path == "/v1/models/needle-3":
            self._json(200, {"id": "needle-3", "object": "model", "owned_by": "cactus", "depth": 20, "variants": []})
        elif self.path == "/v1/models/model-1":
            self._json(200, {"id": "model-1", "object": "model", "owned_by": "me", "name": "smart-home", "depth": 8,
                             "variants": [{"id": "model-1-8", "depth": 8, "is_full": True, "bytes": 5},
                                          {"id": "model-1-4", "depth": 4, "is_full": False, "bytes": 5}]})
        elif self.path.startswith("/v1/models/model-1-") and self.path.endswith("/content"):
            self.send_response(307)
            self.send_header("Location", f"http://{self.headers['Host']}/blob/{self.path.split('/')[3]}")
            self.send_header("Content-Length", "0")
            self.end_headers()
        elif self.path == "/v1/fine_tuning/jobs/ftjob-1":
            s["polls"] += 1
            status = "running" if s["polls"] < 2 else "succeeded"
            self._json(200, {"id": "ftjob-1", "object": "fine_tuning.job", "status": status, "created_at": 0,
                             "completed_steps": min(s["polls"], 5), "total_steps": 5, "max_depth": 8,
                             "fine_tuned_model": "model-1" if status == "succeeded" else None,
                             "evaluations": [{"depth": 8, "validation": {"correct": 9, "total": 10, "excluded": 0},
                                              "test": {"correct": 8, "total": 10, "excluded": 0}}] if status == "succeeded" else []})
        else:
            self._error(404, "not_found")

    def do_POST(self):
        s = self.state
        body = self._body()
        s["requests"].append(("POST", self.path, self.headers.get("Authorization")))
        if self.path == "/v1/files":
            payload = json.loads(body)
            s["reserved"] = payload
            self._json(201, {"id": "0123", "url": f"http://{self.headers['Host']}/upload/0123", "expires_at": "2026-01-01T00:00:00Z"})
        elif self.path == "/v1/files/0123/complete":
            self._json(201, {"id": "file-0123", "object": "file", "filename": s["reserved"]["name"],
                             "bytes": s["reserved"]["bytes"], "created_at": 0, "purpose": "fine-tune"})
        elif self.path == "/v1/fine_tuning/jobs":
            s["submitted"] = json.loads(body)
            if s["submitted"].get("suffix") == "broke":
                self._error(402, "quota_exceeded", param="jobs", url="https://cactuscompute.com/dashboard/usage")
            else:
                self._json(201, {"id": "ftjob-1", "object": "fine_tuning.job", "status": "queued", "created_at": 0,
                                 "max_depth": s["submitted"]["max_depth"]})
        else:
            self._error(404, "not_found")

    def do_PUT(self):
        s = self.state
        body = self._body()
        s["requests"].append(("PUT", self.path, self.headers.get("Authorization")))
        s["uploaded"] = body
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()


@pytest.fixture
def server():
    state = {"requests": [], "polls": 0, "rate_limit_once": False}
    FakePlatform.state = state
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), FakePlatform)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}/v1", state
    finally:
        httpd.shutdown()


def test_missing_key_points_at_the_console(server, monkeypatch):
    from needle.platform import Platform, PlatformError

    base, state = server
    monkeypatch.delenv("NEEDLE_API_KEY", raising=False)
    with pytest.raises(PlatformError) as caught:
        Platform(base_url=base).billing()
    assert caught.value.code == "unauthorized"
    assert "NEEDLE_API_KEY" in str(caught.value)
    assert "dashboard/api-keys" in caught.value.url
    assert state["requests"][-1][2] is None


def test_finetune_uploads_submits_waits_and_downloads(server, tmp_path):
    from needle.platform import Platform

    base, state = server
    train = tmp_path / "train.jsonl"
    train.write_text('{"messages":[],"tools":[]}\n')
    client = Platform(api_key="needle_ft_test", base_url=base)

    job = client.finetune([str(train)], ["file-val"], ["file-test"], suffix="smart-home")
    assert job["id"] == "ftjob-1"
    assert state["reserved"] == {"name": "train.jsonl", "bytes": train.stat().st_size}
    assert state["uploaded"] == train.read_bytes()
    put = next(r for r in state["requests"] if r[0] == "PUT")
    assert put[1] == "/upload/0123" and put[2] is None
    assert state["submitted"] == {"model": "needle-3", "training_files": ["file-0123"],
                                  "validation_files": ["file-val"], "test_files": ["file-test"],
                                  "max_depth": 20, "suffix": "smart-home"}

    seen = []
    done = client.wait(job, poll=0, on_update=lambda r: seen.append(r["status"]))
    assert seen == ["running", "succeeded"]
    assert done["fine_tuned_model"] == "model-1"

    paths = client.download("model-1", str(tmp_path / "out"))
    assert sorted(p.split("/")[-1] for p in paths) == ["smart-home-4L.cact", "smart-home-8L.cact"]
    assert all(open(p, "rb").read() == b"CACT!" for p in paths)
    blob = next(r for r in state["requests"] if r[1].startswith("/blob/"))
    assert blob[2] is None

    only = client.download("model-1", str(tmp_path / "one"), depth=4)
    assert [p.split("/")[-1] for p in only] == ["smart-home-4L.cact"]


def test_errors_carry_code_param_and_url(server, tmp_path):
    from needle.platform import Platform, PlatformError

    base, state = server
    client = Platform(api_key="needle_ft_test", base_url=base)
    with pytest.raises(PlatformError) as caught:
        client.finetune(["file-a"], ["file-b"], ["file-c"], max_depth=8, suffix="broke")
    assert (caught.value.code, caught.value.status, caught.value.param) == ("quota_exceeded", 402, "jobs")
    assert "dashboard/usage" in str(caught.value)

    state["rate_limit_once"] = True
    assert client.billing()["plan"] == "standard"


def test_download_target_routes_hosted_models():
    from needle.cli import _download_target

    assert _download_target("model-abc") == ("hosted", "model-abc")


def test_tool_form_conversions():
    from needle.platform import flat_tools, openai_tools

    flat = [{"name": "set_lights", "parameters": {"type": "object", "properties": {}}}]
    wrapped = openai_tools(flat)
    assert wrapped == [{"type": "function", "function": flat[0]}]
    assert flat_tools(wrapped) == flat
    assert openai_tools(wrapped) == wrapped


def test_local_finetune_reads_chat_format(tmp_path):
    from needle.model.finetune import from_chat, read_examples

    chat = {"messages": [{"role": "system", "content": "date: 2026-09-19"},
                         {"role": "user", "content": "turn the kitchen lights off"},
                         {"role": "assistant", "content": "", "tool_calls": [
                             {"id": "call_1", "type": "function",
                              "function": {"name": "set_lights", "arguments": "{\"room\":\"kitchen\",\"state\":\"off\"}"}}]}],
            "tools": [{"type": "function", "function": {"name": "set_lights", "parameters": {"type": "object"}}}]}
    example = from_chat(chat)
    assert example == {"query": "turn the kitchen lights off", "system": "date: 2026-09-19",
                       "tools": [{"name": "set_lights", "parameters": {"type": "object"}}],
                       "answers": [{"name": "set_lights", "arguments": {"room": "kitchen", "state": "off"}}]}
    silent = {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "Hello."}], "tools": []}
    assert from_chat(silent)["answers"] == []
    multi = {"messages": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"},
                          {"role": "user", "content": "c"}, {"role": "assistant", "content": "d"}], "tools": []}
    assert from_chat(multi) is None

    path = tmp_path / "mixed.jsonl"
    path.write_text("\n".join(json.dumps(x) for x in (chat, silent, multi,
                                                     {"query": "q", "tools": [], "answers": []})) + "\n")
    rows = list(read_examples(str(path)))
    assert [r["query"] for r in rows] == ["turn the kitchen lights off", "hi", "q"]


def test_confidence_head_detected_from_manifest(tmp_path, fake_cact):
    import needle

    fake_cact(tmp_path / "platform.cact", [1, 2])
    fake_cact(tmp_path / "local.cact", [1])
    fake_cact(tmp_path / "bare.cact", [])
    assert needle._confidence_head_present(str(tmp_path / "platform.cact")) is True
    assert needle._confidence_head_present(str(tmp_path / "local.cact")) is False
    assert needle._confidence_head_present(str(tmp_path / "bare.cact")) is False


def _locale_open(encoding):
    """open() as it behaves where the locale's encoding is `encoding` (Windows'
    ANSI code page): a text-mode open without an encoding decodes with it."""
    import builtins

    def locale_open(file, mode="r", *args, **kwargs):
        if "b" not in mode and not args and kwargs.get("encoding") is None:
            kwargs["encoding"] = encoding
        return builtins.open(file, mode, *args, **kwargs)
    return locale_open


@pytest.mark.parametrize("locale_encoding", ["cp1252", "cp949"])
def test_data_and_tool_files_are_utf8_whatever_the_locale(tmp_path, monkeypatch, locale_encoding):
    # The JSON these commands read is UTF-8, but a text-mode open() without an
    # encoding decodes with the locale's. On Windows "café" came back as
    # "cafÃ©" (cp1252) or "caf챕" (cp949) with no error, and an emoji stopped
    # the read with UnicodeDecodeError.
    import types

    import needle.model.finetune as finetune
    import needle.platform as platform

    for module in (finetune, platform):
        monkeypatch.setattr(module, "open", _locale_open(locale_encoding), raising=False)
    query = "set the café lights to 22°C and play 🎵"
    tools = [{"name": "set_lights", "description": "Réglez la lumière à 22°C",
              "parameters": {"type": "object", "properties": {}}}]
    tools_path = tmp_path / "tools.json"
    tools_path.write_text(json.dumps(tools, ensure_ascii=False), encoding="utf-8")
    data = tmp_path / "data.jsonl"
    data.write_text(json.dumps({"query": query, "tools": tools, "answers": []}, ensure_ascii=False) + "\n",
                    encoding="utf-8")

    assert [row["query"] for row in finetune.read_examples(str(data))] == [query]

    seen = {}

    def generate_dataset(tools, num_samples, **kwargs):
        seen["generate"] = tools
        return [{"query": query, "tools": tools, "answers": []}]
    monkeypatch.setattr(finetune, "generate_dataset", generate_dataset)
    out = tmp_path / "generated.jsonl"
    finetune.generate_main(types.SimpleNamespace(
        model=None, workers=1, tools=str(tools_path), output=str(out), num_samples=1,
        batch_size=1, augment=None))
    assert seen["generate"] == tools
    assert [row["query"] for row in finetune.read_examples(str(out))] == [query]

    class Client:
        def billing(self):
            return {}

        def generate(self, tools, examples, **kwargs):
            seen["platform"] = tools
            return {"id": "gen-1"}
    platform._cmd_generate(Client(), types.SimpleNamespace(
        tools=str(tools_path), examples=10, description=None, message=None, suffix=None,
        no_wait=True))
    assert seen["platform"] == tools
