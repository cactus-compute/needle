import json as _json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
from email.parser import BytesParser
from email.policy import default as _email_policy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_STATIC_DIR = Path(__file__).parent / "static"
_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
}
_DEFAULT_MAX_GEN_LEN = 512
_MAX_MAX_GEN_LEN = 4096
_MAX_JSON_BYTES = 1024 * 1024
_MAX_UPLOAD_BYTES = 512 * 1024 * 1024
_SOCKET_TIMEOUT = 30

_model = None
_params = None
_tokenizer = None
_current_model = ""
_current_model_path = None
_lock = threading.Lock()
_finetune_lock = threading.Lock()
_finetune_status = {
    "running": False, "step": "", "log": [], "checkpoint": None,
    "validation": None, "eval_results": None, "base_eval": None,
}


def _is_local_request(client_ip):
    return client_ip in {"127.0.0.1", "::1", "::ffff:127.0.0.1"}


def _is_same_origin(handler):
    """Reject cross-origin POST requests (CSRF protection).

    Browsers always send Origin on cross-origin POSTs.  If Origin is present
    it must match the Host header; if absent we allow (non-browser client).
    """
    origin = handler.headers.get("Origin")
    if origin is None:
        return True
    try:
        parsed = urllib.parse.urlparse(origin)
        origin_host = parsed.hostname or ""
        default_port = 443 if parsed.scheme == "https" else 80
        origin_port = parsed.port or default_port
        host_header = handler.headers.get("Host", "")
        if ":" in host_header:
            h_host, h_port = host_header.rsplit(":", 1)
            h_port = int(h_port)
        else:
            h_host, h_port = host_header, default_port
        return origin_host == h_host and origin_port == h_port
    except Exception:
        return False


def _project_root():
    return Path(__file__).parent.parent.parent


def _checkpoints_dir():
    return _project_root() / "checkpoints"


def _snapshot_finetune_status():
    with _finetune_lock:
        return {
            "running": _finetune_status["running"],
            "step": _finetune_status["step"],
            "log": list(_finetune_status["log"]),
            "checkpoint": _finetune_status["checkpoint"],
            "validation": _finetune_status["validation"],
            "eval_results": _finetune_status["eval_results"],
            "base_eval": _finetune_status["base_eval"],
        }


def _set_finetune_status(**updates):
    with _finetune_lock:
        _finetune_status.update(updates)


_MAX_FINETUNE_LOG_LINES = 2000


def _append_finetune_log(msg):
    with _finetune_lock:
        log = _finetune_status["log"]
        log.append(msg)
        if len(log) > _MAX_FINETUNE_LOG_LINES:
            del log[: len(log) - _MAX_FINETUNE_LOG_LINES]
    print(f"[finetune] {msg}", file=sys.stderr)



def _read_request_body(handler, max_bytes):
    try:
        length = int(handler.headers.get("Content-Length", "0"))
    except ValueError:
        raise ValueError("invalid Content-Length")
    if length <= 0:
        raise ValueError("empty request body")
    if length > max_bytes:
        raise ValueError(f"request body too large (max {max_bytes} bytes)")
    return handler.rfile.read(length)


def _read_json_request(handler, max_bytes=_MAX_JSON_BYTES):
    try:
        raw = _read_request_body(handler, max_bytes)
        body = _json.loads(raw)
    except _json.JSONDecodeError as exc:
        raise ValueError("invalid JSON") from exc
    if not isinstance(body, dict):
        raise ValueError("JSON body must be an object")
    return body


def _parse_bool(value, default):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    raise ValueError("boolean field is invalid")


def _clamp_int(value, default, minimum, maximum, field_name):
    if value in (None, ""):
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer") from exc
    return max(minimum, min(parsed, maximum))


def _normalize_tools_json(tools):
    if tools is None or tools == "":
        return "[]"
    if isinstance(tools, str):
        try:
            parsed = _json.loads(tools)
        except _json.JSONDecodeError as exc:
            raise ValueError("tools must be valid JSON") from exc
    elif isinstance(tools, list):
        parsed = tools
    else:
        raise ValueError("tools must be a JSON string or array")
    if not isinstance(parsed, list):
        raise ValueError("tools must be a JSON array")
    return _json.dumps(parsed, separators=(",", ":"))


def _parse_generate_request(body):
    query = body.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query is required")
    tools = _normalize_tools_json(body.get("tools", "[]"))
    seed = _clamp_int(body.get("seed", 0), 0, 0, 2**31 - 1, "seed")
    max_gen_len = _clamp_int(
        body.get("max_gen_len", _DEFAULT_MAX_GEN_LEN),
        _DEFAULT_MAX_GEN_LEN,
        1,
        _MAX_MAX_GEN_LEN,
        "max_gen_len",
    )
    constrained = _parse_bool(body.get("constrained", True), True)
    return query.strip(), tools, seed, max_gen_len, constrained


def _parse_finetune_request(body):
    tools = _normalize_tools_json(body.get("tools", "[]"))
    parsed_tools = _json.loads(tools)
    if not parsed_tools:
        raise ValueError("tools must be a non-empty JSON array")
    for i, t in enumerate(parsed_tools):
        if not isinstance(t, dict):
            raise ValueError(f"tool[{i}] must be an object")
        if "name" not in t:
            if "function" in t:
                raise ValueError(
                    'Wrong format: tools use OpenAI format ({"type":"function","function":{...}}). '
                    'Needle requires [{"name":"...","description":"...","parameters":{...}}]'
                )
            raise ValueError(f"tool[{i}] missing 'name' field")
        if "description" not in t:
            raise ValueError(f"tool[{i}] missing 'description' field")
    api_key = body.get("api_key")
    if not isinstance(api_key, str) or not api_key.strip():
        raise ValueError("Gemini API key is required")
    return tools, api_key.strip()


def _stream_upload_to_file(handler, max_bytes, target_dir):
    """Parse multipart upload and stream the file part directly to disk.

    Returns (filename, target_path) on success.  Raises ValueError on bad input.
    """
    content_type = handler.headers.get("Content-Type", "")
    if "multipart/form-data" not in content_type:
        raise ValueError("expected multipart/form-data")
    if "boundary=" not in content_type:
        raise ValueError("missing multipart boundary")
    try:
        length = int(handler.headers.get("Content-Length", "0"))
    except ValueError:
        raise ValueError("invalid Content-Length")
    if length <= 0:
        raise ValueError("empty request body")
    if length > max_bytes:
        raise ValueError(f"request body too large (max {max_bytes} bytes)")

    body = handler.rfile.read(length)
    header = f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("utf-8")
    message = BytesParser(policy=_email_policy).parsebytes(header + body)
    del body  # free raw body (~512 MB)
    if not message.is_multipart():
        raise ValueError("expected multipart/form-data")

    filename = None
    payload = None
    for part in message.iter_parts():
        if part.get_param("name", header="content-disposition") != "file":
            continue
        filename = part.get_filename()
        payload = part.get_payload(decode=True)
        break
    del message  # free parsed MIME tree
    if not filename or payload is None:
        raise ValueError("expected a .pkl file")
    if not filename.endswith(".pkl"):
        raise ValueError("expected a .pkl file")
    target_dir.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", Path(filename).name)
    target = target_dir / safe_name
    with open(target, "wb") as f:
        f.write(payload)
    del payload  # free file data
    return filename, target


_datagen_lock = threading.Lock()


def _generate_custom_data(tools_json, api_key, num_samples, data_file):
    import json
    from ..dataset import generate as gd

    tools = json.loads(tools_json)

    def _pick_tools(_rng, force_empty=False, few_tools=False):
        if force_empty:
            return []
        return [dict(t) for t in tools]

    def _synthesize_tools(*args, **kwargs):
        return None

    with _datagen_lock:
        old_api_key = os.environ.get("GEMINI_API_KEY")
        old_pick_tools = gd._pick_tools
        old_synthesize_tools = gd._synthesize_tools
        old_overlap_pairs = gd._OVERLAP_PAIRS
        old_languages = gd.LANGUAGES
        try:
            os.environ["GEMINI_API_KEY"] = api_key
            gd._pick_tools = _pick_tools
            gd._synthesize_tools = _synthesize_tools
            gd._OVERLAP_PAIRS = []
            gd.LANGUAGES = ["English"]
            client_pool = gd.ClientPool(gd.make_clients())
            examples = gd.generate_all(
                num_samples,
                workers=4,
                batch_size=25,
                model=gd.MODEL,
                client_pool=client_pool,
            )
            with open(data_file, "w") as f:
                for ex in examples:
                    f.write(json.dumps(ex) + "\n")
            return len(examples)
        finally:
            gd._pick_tools = old_pick_tools
            gd._synthesize_tools = old_synthesize_tools
            gd._OVERLAP_PAIRS = old_overlap_pairs
            gd.LANGUAGES = old_languages
            if old_api_key is None:
                os.environ.pop("GEMINI_API_KEY", None)
            else:
                os.environ["GEMINI_API_KEY"] = old_api_key


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/":
            path = "/static/index.html"
        if path == "/model-info":
            self._json_response(200, {"name": _current_model})
            return
        if path == "/finetune/status":
            if not _is_local_request(self.client_address[0]):
                self._json_response(403, {"error": "finetune status is only available from localhost"})
                return
            self._json_response(200, _snapshot_finetune_status())
            return
        if path.startswith("/download/"):
            if not _is_local_request(self.client_address[0]):
                self._json_response(403, {"error": "checkpoint download is only allowed from localhost"})
                return
            name = urllib.parse.unquote(path[len("/download/"):])
            if "/" in name or "\\" in name or name.startswith("."):
                self.send_error(400)
                return
            fpath = (_checkpoints_dir() / name).resolve()
            ckpt_dir = _checkpoints_dir().resolve()
            if fpath.is_file() and fpath.is_relative_to(ckpt_dir) and fpath.suffix in (".pkl", ".zip"):
                ctype = "application/zip" if fpath.suffix == ".zip" else "application/octet-stream"
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Disposition", f'attachment; filename="{name}"')
                self.send_header("Content-Length", str(fpath.stat().st_size))
                self.end_headers()
                with open(fpath, "rb") as f:
                    while chunk := f.read(65536):
                        self.wfile.write(chunk)
                return
        if path.startswith("/static/"):
            name = path[len("/static/"):]
            fpath = (_STATIC_DIR / name).resolve()
            if fpath.is_file() and fpath.is_relative_to(_STATIC_DIR.resolve()):
                self.send_response(200)
                self.send_header("Content-Type", _CONTENT_TYPES.get(fpath.suffix, "application/octet-stream"))
                self.end_headers()
                self.wfile.write(fpath.read_bytes())
                return
        self.send_error(404)

    def do_POST(self):
        if not _is_same_origin(self):
            self._json_response(403, {"error": "cross-origin requests are not allowed"})
            return
        if self.path == "/generate":
            self._handle_generate()
        elif self.path == "/finetune":
            self._handle_finetune()
        elif self.path == "/load-model":
            self._handle_load_model()
        else:
            self.send_error(404)

    def _handle_generate(self):
        try:
            body = _read_json_request(self)
            query, tools, seed, max_gen_len, constrained = _parse_generate_request(body)
        except ValueError as exc:
            self._json_response(400, {"error": str(exc)})
            return

        result, confidence, error = _run_generate(query, tools, seed, max_gen_len, constrained)
        if error:
            self._json_response(500, {"error": error})
        else:
            self._json_response(200, {"result": result, "confidence": confidence})

    def _handle_finetune(self):
        if not _is_local_request(self.client_address[0]):
            self._json_response(403, {"error": "finetune is only allowed from localhost"})
            return
        try:
            body = _read_json_request(self)
            tools, api_key = _parse_finetune_request(body)
        except ValueError as exc:
            self._json_response(400, {"error": str(exc)})
            return
        if not _current_model_path:
            self._json_response(400, {"error": "Load a model before finetuning"})
            return
        if not _start_finetune(tools, api_key):
            self._json_response(409, {"error": "finetune already running"})
            return
        self._json_response(200, {"status": "started"})

    def _handle_load_model(self):
        if not _is_local_request(self.client_address[0]):
            self._json_response(403, {"error": "model upload is only allowed from localhost"})
            return
        upload_dir = _checkpoints_dir() / "_uploaded"
        try:
            filename, target = _stream_upload_to_file(self, _MAX_UPLOAD_BYTES, upload_dir)
        except ValueError as exc:
            self._json_response(400, {"error": str(exc)})
            return
        try:
            _load_checkpoint(str(target), display_name=filename)
            self._json_response(200, {"name": filename})
        except Exception as exc:
            target.unlink(missing_ok=True)
            self._json_response(500, {"error": "Failed to load checkpoint"})

    def _json_response(self, code, data):
        payload = _json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt, *args):
        pass


def _run_generate(query, tools, seed, max_gen_len, constrained):
    from ..model.run import generate

    with _lock:
        if _model is None or _params is None or _tokenizer is None:
            return None, "model is not loaded"
        try:
            result, confidence = generate(
                _model,
                _params,
                _tokenizer,
                query,
                tools=tools,
                max_gen_len=max_gen_len,
                seed=seed,
                stream=False,
                constrained=constrained,
                return_confidence=True,
            )
            return result, confidence, None
        except Exception as exc:
            print(f"[generate] {exc}", file=sys.stderr)
            return None, None, "Generation failed"


def _load_checkpoint(path, display_name=None):
    global _model, _params, _tokenizer, _current_model, _current_model_path
    from ..dataset.dataset import get_tokenizer
    from ..model.architecture import SimpleAttentionNetwork
    from ..model.run import load_checkpoint

    with _lock:
        _params, config = load_checkpoint(path)
        _model = SimpleAttentionNetwork(config)
        _tokenizer = get_tokenizer()
        _current_model = display_name or Path(path).name
        _current_model_path = path
        print(f"Loaded: {_current_model}", file=sys.stderr)


def _validate_training_data(data_file_path):
    """Validate JSONL training data and return a report dict."""
    import collections
    examples = []
    warnings = []
    with open(data_file_path) as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            try:
                ex = _json.loads(line)
            except _json.JSONDecodeError:
                warnings.append(f"Line {i+1}: invalid JSON")
                continue
            for key in ("query", "tools", "answers"):
                if key not in ex:
                    warnings.append(f"Line {i+1}: missing '{key}'")
            examples.append(ex)

    tool_counts = collections.Counter()
    for ex in examples:
        try:
            answers = _json.loads(ex.get("answers", "[]"))
            for a in answers:
                if isinstance(a, dict) and "name" in a:
                    tool_counts[a["name"]] += 1
        except (_json.JSONDecodeError, TypeError):
            pass

    seen = set()
    dup_count = 0
    for ex in examples:
        key = (ex.get("query", ""), ex.get("answers", ""))
        if key in seen:
            dup_count += 1
        seen.add(key)

    counts = list(tool_counts.values()) or [0]
    return {
        "total": len(examples),
        "per_tool": dict(tool_counts),
        "duplicates": dup_count,
        "min_per_tool": min(counts),
        "max_per_tool": max(counts),
        "warnings": warnings[:20],
    }


_SAMPLES_PER_TOOL = 120  # 100 train + 10 val + 10 test
_EPOCHS = 1


def _start_finetune(tools_json, api_key):
    with _finetune_lock:
        if _finetune_status["running"]:
            return False
        _finetune_status["running"] = True
        _finetune_status["step"] = "starting"
        _finetune_status["log"] = []
        _finetune_status["checkpoint"] = None
        _finetune_status["validation"] = None
        _finetune_status["eval_results"] = None
        _finetune_status["base_eval"] = None

    def _run():
        python = sys.executable
        work_dir = Path(tempfile.mkdtemp(prefix="needle-ui-finetune-", dir=str(_project_root())))
        data_file = work_dir / "data.jsonl"
        cache_dir = work_dir / "cache"

        try:
            num_tools = len(_json.loads(tools_json))
            num_samples = _SAMPLES_PER_TOOL * num_tools

            _set_finetune_status(step="generating data")
            _append_finetune_log(f"Generating {_SAMPLES_PER_TOOL} samples/tool for {num_tools} tools...")
            generated = _generate_custom_data(tools_json, api_key, num_samples, data_file)
            if generated < 3:
                raise RuntimeError("generated fewer than 3 examples; need more data for train/val/test")
            _append_finetune_log(f"Generated {generated} samples.")

            # Data validation
            # Validation runs as part of data generation step
            _append_finetune_log("Validating training data...")
            validation = _validate_training_data(str(data_file))
            _set_finetune_status(validation=validation)
            _append_finetune_log(
                f"Validated: {validation['total']} examples, "
                f"{len(validation['per_tool'])} tools, "
                f"{validation['duplicates']} duplicates"
            )
            if validation["warnings"]:
                for w in validation["warnings"][:5]:
                    _append_finetune_log(f"  WARNING: {w}")

            # Training subprocess
            _set_finetune_status(step="training")
            checkpoint_arg = []
            ckpt_name = "scratch"
            ckpt_dir = _checkpoints_dir()
            existing_pkls = set()
            if ckpt_dir.exists():
                existing_pkls = set(p.name for p in ckpt_dir.glob("*.pkl"))
            if _current_model_path:
                checkpoint_arg = ["--checkpoint", _current_model_path]
                ckpt_name = Path(_current_model_path).name
            batch_size = min(32, generated)
            _append_finetune_log(
                f"Training {_EPOCHS} epochs, batch_size={batch_size}, "
                f"{generated} examples, from {ckpt_name}"
            )

            _set_finetune_status(step="evaluating base")
            last_eval = None
            proc = subprocess.Popen(
                [
                    python, "-u",
                    "-m", "needle.training.finetune",
                    str(data_file),
                    "--epochs", str(_EPOCHS),
                    "--batch-size", str(batch_size),
                    "--checkpoint-dir", str(ckpt_dir),
                    "--cache-dir", str(cache_dir),
                    *checkpoint_arg,
                ],
                cwd=str(_project_root()),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            try:
                for line in proc.stdout:
                    line = line.rstrip("\n")
                    if line.startswith("BASE_EVAL:"):
                        try:
                            _set_finetune_status(base_eval=_json.loads(line[10:]), step="training")
                        except _json.JSONDecodeError:
                            pass
                    elif line.startswith("FINETUNED_EVAL:"):
                        try:
                            last_eval = _json.loads(line[15:])
                            _set_finetune_status(eval_results=last_eval)
                        except _json.JSONDecodeError:
                            pass
                    elif line:
                        if "Evaluating finetuned" in line:
                            _set_finetune_status(step="evaluating finetuned")
                        _append_finetune_log(line)
                proc.wait(timeout=7200)
            except subprocess.TimeoutExpired:
                proc.kill()
                _append_finetune_log("Training timed out (2h limit).")
                _set_finetune_status(step="failed")
                return

            if proc.returncode != 0:
                _append_finetune_log("Training failed (see log above).")
                _set_finetune_status(step="failed")
                return
            _append_finetune_log("Training complete.")

            # Store final eval results
            if last_eval:
                _set_finetune_status(eval_results=last_eval)

            new_files = [p for p in ckpt_dir.glob("*.pkl") if p.name not in existing_pkls]
            if not new_files:
                _append_finetune_log("No new checkpoint produced.")
                _set_finetune_status(step="done")
                return

            newest = max(new_files, key=lambda p: p.stat().st_mtime)
            ts = time.strftime("%Y%m%d_%H%M%S")
            final_name = f"needle_finetuned_{ts}.pkl"
            final_path = ckpt_dir / final_name
            newest.replace(final_path)

            # Assemble download bundle (same per-tool split as finetune)
            import zipfile
            from ..training.finetune import _per_tool_split

            with open(data_file) as _df:
                all_examples = [_json.loads(ln) for ln in _df if ln.strip()]
            train_examples, val_examples, test_examples = _per_tool_split(all_examples)

            with _finetune_lock:
                base_eval = _finetune_status.get("base_eval")

            per_tool_report = {}
            if base_eval and base_eval.get("per_tool"):
                for t, bm in base_eval["per_tool"].items():
                    per_tool_report[t] = {"base": round(bm["correct"] / max(bm["total"], 1), 3)}
            if last_eval and last_eval.get("per_tool"):
                for t, fm in last_eval["per_tool"].items():
                    entry = per_tool_report.setdefault(t, {})
                    entry["finetuned"] = round(fm["correct"] / max(fm["total"], 1), 3)

            eval_report = {
                "model": ckpt_name,
                "finetuned_checkpoint": final_name,
                "training": {"examples": len(train_examples), "tools": num_tools, "epochs": _EPOCHS},
                "validation": {"examples": len(val_examples)},
                "test": {"examples": len(test_examples), "duplicates": validation.get("duplicates", 0)},
                "base": {k: base_eval.get(k) for k in ("call_f1", "name_f1", "exact_match", "parse_rate", "args_acc")} if base_eval else None,
                "finetuned": {k: last_eval.get(k) for k in ("call_f1", "name_f1", "exact_match", "parse_rate", "args_acc")} if last_eval else None,
                "per_tool": per_tool_report,
            }

            def _pct(v):
                return f"{v * 100:.1f}%" if v is not None else "n/a"

            b = eval_report.get("base") or {}
            f_ = eval_report.get("finetuned") or {}
            readme = (
                f"# Needle Finetuned Model\n\n"
                f"## Training\n"
                f"- Base model: {ckpt_name}\n"
                f"- Epochs: {_EPOCHS}\n"
                f"- Training examples: {len(train_examples)}\n"
                f"- Validation examples: {len(val_examples)} (used for checkpoint selection)\n"
                f"- Test examples: {len(test_examples)} (held out, used for eval)\n"
                f"- Tools: {num_tools}\n\n"
                f"## Evaluation (on held-out test set)\n\n"
                f"| Metric | Base | Finetuned |\n"
                f"|--------|------|----------|\n"
                f"| Call F1 | {_pct(b.get('call_f1'))} | {_pct(f_.get('call_f1'))} |\n"
                f"| Name F1 | {_pct(b.get('name_f1'))} | {_pct(f_.get('name_f1'))} |\n"
                f"| Exact Match | {_pct(b.get('exact_match'))} | {_pct(f_.get('exact_match'))} |\n"
                f"| Parse Rate | {_pct(b.get('parse_rate'))} | {_pct(f_.get('parse_rate'))} |\n"
                f"| Args Accuracy | {_pct(b.get('args_acc'))} | {_pct(f_.get('args_acc'))} |\n\n"
                f"## Contents\n"
                f"- `checkpoint.pkl` — Finetuned model weights\n"
                f"- `tools.json` — Tool definitions\n"
                f"- `train.jsonl` — Training data ({len(train_examples)} examples)\n"
                f"- `val.jsonl` — Validation data ({len(val_examples)} examples)\n"
                f"- `test.jsonl` — Test data ({len(test_examples)} examples)\n"
                f"- `eval_report.json` — Machine-readable evaluation\n"
            )

            bundle_name = f"needle_finetuned_{ts}.zip"
            bundle_path = ckpt_dir / bundle_name
            with zipfile.ZipFile(str(bundle_path), "w", zipfile.ZIP_DEFLATED) as zf:
                zf.write(str(final_path), "checkpoint.pkl")
                zf.writestr("tools.json", tools_json)
                zf.writestr("train.jsonl", "\n".join(_json.dumps(ex) for ex in train_examples) + "\n")
                zf.writestr("val.jsonl", "\n".join(_json.dumps(ex) for ex in val_examples) + "\n")
                zf.writestr("test.jsonl", "\n".join(_json.dumps(ex) for ex in test_examples) + "\n")
                zf.writestr("eval_report.json", _json.dumps(eval_report, indent=2))
                zf.writestr("README.md", readme)

            _set_finetune_status(checkpoint=bundle_name)
            _append_finetune_log(f"Bundle ready: {bundle_name}")
            _set_finetune_status(step="done")
        except Exception as exc:
            _append_finetune_log(f"Error: {exc}")
            _set_finetune_status(step="failed")
        finally:
            _set_finetune_status(running=False)
            shutil.rmtree(work_dir, ignore_errors=True)

    threading.Thread(target=_run, daemon=True).start()
    return True


_HF_MODEL_REPO = "Cactus-Compute/needle"
_HF_MODEL_FILE = "needle.pkl"


def _resolve_checkpoint(checkpoint_arg):
    """Resolve checkpoint path: always download from HuggingFace to ensure freshness."""
    from huggingface_hub import hf_hub_download
    local_dir = "checkpoints"
    os.makedirs(local_dir, exist_ok=True)
    filename = os.path.basename(checkpoint_arg) if checkpoint_arg else _HF_MODEL_FILE
    repo = _HF_MODEL_REPO
    print(f"Downloading {filename} from {repo}...", file=sys.stderr)
    path = hf_hub_download(
        repo_id=repo,
        filename=filename,
        repo_type="model",
        local_dir=local_dir,
        force_download=True,
    )
    print(f"Downloaded to {path}", file=sys.stderr)
    return path


def main(args):
    global _model, _params, _tokenizer, _current_model, _current_model_path
    from ..dataset.dataset import get_tokenizer
    from ..model.architecture import SimpleAttentionNetwork
    from ..model.run import load_checkpoint

    checkpoint_path = _resolve_checkpoint(args.checkpoint)
    print(f"Loading checkpoint: {checkpoint_path}", file=sys.stderr)
    _params, config = load_checkpoint(checkpoint_path)
    _model = SimpleAttentionNetwork(config)
    _tokenizer = get_tokenizer()
    _current_model = Path(checkpoint_path).name
    _current_model_path = checkpoint_path

    param_count = sum(x.size for x in __import__("jax").tree.leaves(_params))
    print(f"Model parameters: {param_count:,}", file=sys.stderr)

    server = ThreadingHTTPServer((args.host, args.port), _Handler)
    server.timeout = _SOCKET_TIMEOUT
    server.request_queue_size = 16
    print(f"Needle UI: http://{args.host}:{args.port}", file=sys.stderr)
    server.serve_forever()
