import json
import os
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_DIR = Path(__file__).parent
_STATIC = {"/": "index.html", "/index.html": "index.html",
           "/app.js": "app.js", "/style.css": "style.css"}
_CTYPE = {".html": "text/html; charset=utf-8",
          ".js": "application/javascript; charset=utf-8",
          ".css": "text/css; charset=utf-8"}
_DOWNLOADS = Path(tempfile.mkdtemp(prefix="needle-playground-"))


class Engine:
    def __init__(self, weights=None):
        self.weights = weights
        self.name = os.path.basename(weights) if weights else "needle-2 (base)"
        self.lock = threading.Lock()
        self.tools_json = None
        self.agent = None

    def load(self):
        from .. import Needle
        self.agent = Needle(tools="[]", weights=self.weights)
        self.tools_json = "[]"

    def complete(self, tools_json, query):
        from .. import Needle, _lib
        with self.lock:
            if self.agent is None or tools_json != self.tools_json:
                self.agent = Needle(tools=tools_json, weights=self.weights)
                self.tools_json = tools_json
            else:
                _lib().needle_reset()
            return self.agent.complete(query)

    def reset(self):
        from .. import _lib
        with self.lock:
            if self.agent is not None:
                _lib().needle_reset()
            self.agent = None
            self.tools_json = None

    def load_weights(self, path):
        with self.lock:
            self.weights = path
            self.name = os.path.basename(path)
            self.agent = None
            self.tools_json = None
        self.load()


_FT = {"running": False, "step": "", "log": [], "checkpoint": None, "error": None}


def _log(msg):
    _FT["log"].append(msg)
    if len(_FT["log"]) > 100:
        del _FT["log"][:-100]


def _finetune_worker(tools_json, api_key, samples, engine):
    import types
    try:
        _FT.update(running=True, step="generating data", log=[], checkpoint=None, error=None)
        from ..model.finetune import generate_dataset, finetune_local, build_main, DEFAULT_BASE

        tools = json.loads(tools_json)
        rows = generate_dataset(
            tools, samples, api_key=api_key,
            progress=lambda done, total: _log(f"generated {done}/{total}"))
        data_path = str(_DOWNLOADS / "needle_playground_data.jsonl")
        with open(data_path, "w") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")

        _FT["step"] = "training"
        adapter = str(_DOWNLOADS / "needle_playground_lora.pkl")
        finetune_local(types.SimpleNamespace(
            jsonl_path=data_path, checkpoint=None, epochs=3, batch_size=16, lr=1e-4,
            lora_rank=16, lora_alpha=32.0, max_len=1024, generate=0, model=None,
            checkpoint_dir=str(_DOWNLOADS), out=adapter), progress=_log)

        _FT["step"] = "building"
        out = str(_DOWNLOADS / "needle_tuned.cact")
        build_main(types.SimpleNamespace(checkpoint=DEFAULT_BASE, lora=adapter,
                                         out=out, upload=False, bits=None))
        engine.load_weights(out)
        _FT.update(running=False, step="done", checkpoint=os.path.basename(out))
    except Exception as exc:
        _FT.update(running=False, step="failed", error=str(exc))
        _FT["log"].append(str(exc))


class _Handler(BaseHTTPRequestHandler):
    engine = None

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, (bytes, bytearray)) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _json_body(self):
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length) or b"{}")

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in _STATIC:
            f = _DIR / _STATIC[path]
            self._send(200, f.read_bytes(), _CTYPE[f.suffix])
        elif path == "/model":
            self._send(200, json.dumps({"name": self.engine.name}))
        elif path == "/finetune/status":
            self._send(200, json.dumps(_FT))
        elif path.startswith("/download/"):
            name = os.path.basename(path[len("/download/"):])
            f = _DOWNLOADS / name
            if f.exists():
                self._send(200, f.read_bytes(), "application/octet-stream")
            else:
                self._send(404, b"not found", "text/plain")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        try:
            if self.path == "/complete":
                body = self._json_body()
                tools = body.get("tools", [])
                tools_json = tools if isinstance(tools, str) else json.dumps(tools)
                result = self.engine.complete(tools_json, body.get("query", ""))
                self._send(200, json.dumps(result))
            elif self.path == "/reset":
                self.engine.reset()
                self._send(200, json.dumps({"ok": True}))
            elif self.path == "/load-model":
                name = os.path.basename(self.headers.get("X-Filename", "model.cact"))
                length = int(self.headers.get("Content-Length", 0))
                dest = _DOWNLOADS / name
                dest.write_bytes(self.rfile.read(length))
                self.engine.load_weights(str(dest))
                self._send(200, json.dumps({"name": self.engine.name}))
            elif self.path == "/finetune":
                if _FT["running"]:
                    self._send(200, json.dumps({"error": "a finetune is already running"}))
                    return
                body = self._json_body()
                api_key = (body.get("api_key") or "").strip()
                if not api_key:
                    self._send(200, json.dumps({"error": "OpenRouter API key is required"}))
                    return
                tools = body.get("tools", "[]")
                tools_json = tools if isinstance(tools, str) else json.dumps(tools)
                samples = int(body.get("samples", 200))
                threading.Thread(target=_finetune_worker,
                                 args=(tools_json, api_key, samples, self.engine),
                                 daemon=True).start()
                self._send(200, json.dumps({"ok": True}))
            else:
                self._send(404, b"not found", "text/plain")
        except Exception as exc:
            self._send(200, json.dumps({"error": str(exc)}))

    def log_message(self, *args):
        pass


def main(args):
    engine = Engine(weights=getattr(args, "weights", None))
    print("needle playground: downloading and initializing the model...", flush=True)
    engine.load()
    _Handler.engine = engine
    server = ThreadingHTTPServer((args.host, args.port), _Handler)
    print(f"needle playground ready: http://{args.host}:{args.port}  ({engine.name})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
