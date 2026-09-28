from __future__ import annotations

import http.client
import json
import os
import shutil
import socket
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request

BASE_URL = "https://cactuscompute.com/v1"
KEYS_URL = "https://cactuscompute.com/dashboard/api-keys"
JOBS_URL = "https://cactuscompute.com/dashboard/jobs"
BASE_MODEL = "needle-3"
TERMINAL = ("succeeded", "failed", "cancelled")


class PlatformError(RuntimeError):
    def __init__(self, code, message, status=0, param=None, url=None, retry_after=0):
        super().__init__(message)
        self.code = code
        self.status = status
        self.param = param
        self.url = url
        self.retry_after = retry_after

    def __str__(self):
        text = f"{self.code}: {self.args[0]}"
        if self.param:
            text += f" (param: {self.param})"
        if self.url:
            text += f" -> {self.url}"
        return text


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _ssl_context():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def flat_tools(tools):
    """OpenAI-form tool definitions to the flat form `needle.Needle(tools=...)` takes."""
    return [t["function"] if t.get("type") == "function" and "function" in t else t
            for t in tools]


def openai_tools(tools):
    """Flat tool definitions to the OpenAI form the platform takes."""
    return [t if t.get("type") == "function" else {"type": "function", "function": t}
            for t in tools]


class Platform:
    """Client for the hosted fine-tuning API at cactuscompute.com/v1.

    The key comes from `api_key` or `NEEDLE_API_KEY`; create one in the console
    under API Keys. Every method raises `PlatformError` with the API's stable
    error `code` on failure.
    """

    def __init__(self, api_key=None, base_url=BASE_URL, timeout=60, retries=3):
        self.api_key = api_key or os.environ.get("NEEDLE_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = retries
        https = urllib.request.HTTPSHandler(context=_ssl_context())
        self._opener = urllib.request.build_opener(_NoRedirect, https)
        self._follower = urllib.request.build_opener(https)

    def _send(self, method, url, body=None, headers=None, auth=True):
        headers = dict(headers or {})
        if auth and self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        data = None
        if body is not None and not isinstance(body, (bytes, bytearray)):
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        elif body is not None:
            data = bytes(body)
        request = urllib.request.Request(url, data=data, method=method, headers=headers)
        for attempt in range(self.retries + 1):
            try:
                with self._opener.open(request, timeout=self.timeout) as response:
                    return response.status, dict(response.headers), response.read()
            except urllib.error.HTTPError as error:
                payload = error.read()
                if error.code in (301, 302, 303, 307, 308):
                    return error.code, dict(error.headers), payload
                code, message, param, link = _parse_error(payload, error.code)
                if error.code == 401 and not self.api_key:
                    message, link = "set NEEDLE_API_KEY to a key from the console", link or KEYS_URL
                retry_after = _seconds(error.headers.get("Retry-After"))
                if error.code == 429 and code == "rate_limited" and attempt < self.retries:
                    time.sleep(max(retry_after, 1))
                    continue
                raise PlatformError(code, message, error.code, param, link, retry_after) from None
            except urllib.error.URLError as error:
                raise PlatformError("network_error", str(error.reason)) from None
        raise PlatformError("rate_limited", "the API kept answering 429", 429)

    def _call(self, method, path, body=None, params=None):
        url = f"{self.base_url}/{path}"
        if params:
            url += "?" + urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
        status, headers, payload = self._send(method, url, body)
        return json.loads(payload.decode("utf-8")) if payload else {}

    def _signed_link(self, path):
        status, headers, _ = self._send("GET", f"{self.base_url}/{path}")
        link = headers.get("Location") or headers.get("location")
        if status not in (301, 302, 303, 307, 308) or not link:
            raise PlatformError("unexpected_response", f"expected a redirect from {path}, got {status}", status)
        return link

    def _fetch_to(self, link, dest):
        """Download `link` to `dest`. The bytes land in a `.part` file that
        replaces `dest` only once complete, so a download that fails keeps the
        file it would have replaced, and the failure is a PlatformError like
        every other call's."""
        os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
        partial = dest + ".part"
        request = urllib.request.Request(link, method="GET")
        name = os.path.basename(dest)
        try:
            with self._follower.open(request, timeout=self.timeout) as response, open(partial, "wb") as out:
                shutil.copyfileobj(response, out)
                # http.client's read(n) returns what arrived when the server
                # hangs up early, without the IncompleteRead a plain read()
                # raises, so a cut download looks complete unless its length
                # is checked against the one the server announced.
                expected = response.headers.get("Content-Length")
                if expected is not None and out.tell() != int(expected):
                    raise PlatformError("network_error", f"downloading {name} ended after "
                                        f"{out.tell()} of {int(expected)} bytes")
            os.replace(partial, dest)
        except urllib.error.HTTPError as error:
            raise PlatformError(f"http_{error.code}", f"downloading {name} failed with HTTP {error.code}",
                                error.code) from None
        except (urllib.error.URLError, http.client.HTTPException, ConnectionError, socket.timeout) as error:
            raise PlatformError("network_error", f"downloading {name} failed: "
                                f"{getattr(error, 'reason', None) or error!r}") from None
        finally:
            if os.path.exists(partial):
                os.remove(partial)
        return dest

    def plans(self):
        return self._call("GET", "plans")

    def billing(self):
        return self._call("GET", "billing")

    def tool_schemas(self, description):
        return self._call("POST", "tool_schemas", {"description": description})["tools"]

    def files(self, search=None, limit=100):
        return self._call("GET", "files", params={"search": search, "limit": limit})["data"]

    def file(self, file_id):
        return self._call("GET", f"files/{file_id}")

    def delete_file(self, file_id):
        return self._call("DELETE", f"files/{file_id}")

    def upload(self, path, name=None):
        """Upload a .jsonl file in the platform's three steps; returns the file record."""
        size = os.path.getsize(path)
        reservation = self._call("POST", "files", {"name": name or os.path.basename(path), "bytes": size})
        with open(path, "rb") as handle:
            self._send("PUT", reservation["url"], handle.read(),
                       {"Content-Type": "application/octet-stream"}, auth=False)
        return self._call("POST", f"files/{reservation['id']}/complete")

    def download_file(self, file_id, dest):
        return self._fetch_to(self._signed_link(f"files/{file_id}/content"), dest)

    def generate(self, tools, examples, description=None, messages=None, suffix=None):
        body = {"tools": openai_tools(tools), "examples": int(examples)}
        if description:
            body["description"] = description
        if messages:
            body["messages"] = list(messages)
        if suffix:
            body["suffix"] = suffix
        return self._call("POST", "generations", body)

    def generation(self, job_id):
        return self._call("GET", f"generations/{job_id}")

    def generations(self, limit=20):
        return self._call("GET", "generations", params={"limit": limit})["data"]

    def cancel_generation(self, job_id):
        return self._call("POST", f"generations/{job_id}/cancel")

    def _file_ids(self, items):
        ids = []
        for item in items if isinstance(items, (list, tuple)) else [items]:
            if isinstance(item, dict):
                ids.append(item["id"])
            elif isinstance(item, str) and item.startswith("file-"):
                ids.append(item)
            elif isinstance(item, str) and os.path.exists(item):
                ids.append(self.upload(item)["id"])
            else:
                raise PlatformError("invalid_request", f"{item!r} is neither a file id nor a path", 400)
        return ids

    def finetune(self, training_files, validation_files, test_files, max_depth=None, suffix=None):
        """Start a fine-tune. Each group takes paths (uploaded first), file ids, or file records."""
        if max_depth is None:
            max_depth = self.model(BASE_MODEL)["depth"]
        body = {
            "model": BASE_MODEL,
            "training_files": self._file_ids(training_files),
            "validation_files": self._file_ids(validation_files),
            "test_files": self._file_ids(test_files),
            "max_depth": int(max_depth),
        }
        if suffix:
            body["suffix"] = suffix
        return self._call("POST", "fine_tuning/jobs", body)

    def job(self, job_id):
        return self._call("GET", f"fine_tuning/jobs/{job_id}")

    def jobs(self, limit=20):
        return self._call("GET", "fine_tuning/jobs", params={"limit": limit})["data"]

    def cancel(self, job_id):
        return self._call("POST", f"fine_tuning/jobs/{job_id}/cancel")

    def wait(self, job, poll=10, on_update=None):
        """Poll a fine-tune or generation until it ends; returns the record, raises if it failed."""
        job_id = job["id"] if isinstance(job, dict) else job
        fetch = self.generation if (isinstance(job, dict) and job.get("object") == "generation") else self.job
        while True:
            record = fetch(job_id)
            if on_update:
                on_update(record)
            status = record.get("status")
            if status in TERMINAL:
                break
            time.sleep(poll)
        if status == "succeeded":
            return record
        error = record.get("error") or {}
        raise PlatformError(error.get("code") or status, error.get("message") or f"job {job_id} {status}")

    def models(self, search=None, limit=100):
        return self._call("GET", "models", params={"search": search, "limit": limit})["data"]

    def model(self, model_id):
        return self._call("GET", f"models/{model_id}")

    def delete_model(self, model_id):
        return self._call("DELETE", f"models/{model_id}")

    def download(self, model_id, out=".", depth=None):
        """Download a model's .cact files into `out`, every size or one `depth`; returns the paths."""
        record = self.model(model_id)
        stem = record.get("name") or record["id"]
        variants = record.get("variants") or [{"id": record["id"], "depth": record.get("depth")}]
        if depth is not None:
            variants = [v for v in variants if v.get("depth") == int(depth)]
            if not variants:
                raise PlatformError("not_found", f"{model_id} has no {depth}-layer size", 404)
        paths = []
        for variant in variants:
            dest = os.path.join(out, f"{stem}-{variant['depth']}L.cact")
            paths.append(self._fetch_to(self._signed_link(f"models/{variant['id']}/content"), dest))
        return paths


def _seconds(value):
    try:
        return int(float(value or 0))
    except ValueError:
        return 0


def _parse_error(payload, status):
    try:
        error = json.loads(payload.decode("utf-8")).get("error") or {}
    except (ValueError, AttributeError):
        error = {}
    return (error.get("code") or f"http_{status}", error.get("message") or f"HTTP {status}",
            error.get("param"), error.get("url"))


def _print(label, value):
    print(f"  {label:<9} {value}")


def _print_usage(client):
    billing = client.billing()
    limits, usage = billing.get("limits") or {}, billing.get("usage") or {}
    _print("plan", f"{billing.get('plan')}: {usage.get('jobs', 0)}/{limits.get('jobs', '?')} fine-tunes, "
                   f"{usage.get('examples', 0)}/{limits.get('examples', '?')} generated examples this period")


def _print_job(job):
    line = f"{job['id']}  {job.get('status')}"
    if job.get("name"):
        line += f"  {job['name']}"
    if job.get("max_depth"):
        line += f"  max_depth {job['max_depth']}"
    if job.get("error"):
        line += f"  error {job['error'].get('code')}"
    print("  " + line)


def _print_evaluations(job):
    rows = job.get("evaluations") or []
    if not rows:
        return
    print(f"  {'depth':>5}  {'validation':>12}  {'test':>12}")
    for row in sorted(rows, key=lambda r: -r["depth"]):
        v, t = row["validation"], row["test"]
        print(f"  {row['depth']:>4}L  {v['correct']:>5}/{v['total']:<6}  {t['correct']:>5}/{t['total']:<6}")


def _progress(kind):
    seen = {}

    def show(record):
        key = (record.get("status"), record.get("completed_steps"))
        if key in seen:
            return
        seen[key] = True
        status = record.get("status")
        if status == "running" and record.get("total_steps"):
            print(f"  {kind:<9} running, step {record.get('completed_steps', 0)}/{record['total_steps']}", flush=True)
        else:
            print(f"  {kind:<9} {status}", flush=True)
    return show


def _cmd_finetune(client, args):
    _print_usage(client)
    groups = {}
    for label, item in (("train", args.train), ("validation", args.validation), ("test", args.test)):
        if item.startswith("file-"):
            groups[label] = item
        else:
            record = client.upload(item)
            _print("upload", f"{record['id']}  {record['filename']}  {record['bytes'] / 1e6:.2f} MB")
            groups[label] = record["id"]
    job = client.finetune([groups["train"]], [groups["validation"]], [groups["test"]],
                          max_depth=args.max_depth, suffix=args.suffix)
    _print("job", f"{job['id']}  max_depth {job.get('max_depth')}")
    _print("watch", f"{JOBS_URL}/{job['id']}  (Ctrl-C leaves the job running; needle platform jobs {job['id']} --wait --out DIR resumes)")
    if args.no_wait:
        return
    job = client.wait(job, on_update=_progress("job"))
    _print_evaluations(job)
    paths = client.download(job["fine_tuned_model"], args.out, depth=args.depth)
    for path in paths:
        _print("weights", f"{path}  {os.path.getsize(path) / 1e6:.2f} MB")
    _print("next", f"needle.Needle(weights={paths[-1]!r}, tools=[...], auto_date=False)")


def _cmd_generate(client, args):
    _print_usage(client)
    with open(args.tools) as handle:
        tools = json.load(handle)
    job = client.generate(tools, args.examples, description=args.description,
                          messages=args.message, suffix=args.suffix)
    _print("job", f"{job['id']}  {args.examples} examples")
    _print("watch", f"{JOBS_URL}/{job['id']}  (Ctrl-C leaves the job running; needle platform jobs {job['id']} --wait --out DIR resumes)")
    if args.no_wait:
        return
    job = client.wait(job, on_update=_progress("job"))
    for record in job.get("files") or []:
        dest = client.download_file(record["id"], os.path.join(args.out, record["filename"]))
        _print(record["group"], f"{dest}  {os.path.getsize(dest) / 1e6:.2f} MB")
    _print("next", "needle platform finetune <train.jsonl> <validation.jsonl> <test.jsonl>")


def _cmd_jobs(client, args):
    if not args.job_id:
        for job in client.jobs():
            _print_job(job)
        return
    try:
        job = client.job(args.job_id)
    except PlatformError as error:
        if error.status != 404:
            raise
        job = client.generation(args.job_id)
    if args.wait and job.get("status") not in TERMINAL:
        job = client.wait(job, on_update=_progress("job"))
    _print_job(job)
    _print_evaluations(job)
    if job.get("fine_tuned_model"):
        _print("model", job["fine_tuned_model"])
        if args.out:
            for path in client.download(job["fine_tuned_model"], args.out, depth=args.depth):
                _print("weights", f"{path}  {os.path.getsize(path) / 1e6:.2f} MB")
        return
    for record in job.get("files") or []:
        if args.out and job.get("object") == "generation" and job.get("status") == "succeeded":
            dest = client.download_file(record["id"], os.path.join(args.out, record["filename"]))
            _print(record.get("group", "file"), f"{dest}  {os.path.getsize(dest) / 1e6:.2f} MB")
        else:
            _print(record.get("group", "file"), f"{record['id']}  {record['filename']}")


def _cmd_models(client, args):
    if args.model_id:
        record = client.model(args.model_id)
        _print("model", f"{record['id']}  {record.get('name') or ''}  {record.get('depth')} layers")
        for variant in record.get("variants") or []:
            size = f"  {variant['bytes'] / 1e6:.2f} MB" if variant.get("bytes") else ""
            _print(f"{variant['depth']}L", f"{variant['id']}{size}")
        _print("next", f"needle download {record['id']} [--depth N]")
        return
    for record in client.models():
        _print("model", f"{record['id']}  {record.get('name') or ''}  {record.get('depth')} layers")


def _cmd_files(client, args):
    for record in client.files():
        _print("file", f"{record['id']}  {record['filename']}  {record['bytes'] / 1e6:.2f} MB")


def _cmd_billing(client, args):
    billing = client.billing()
    _print("plan", billing.get("plan"))
    for key, limit in (billing.get("limits") or {}).items():
        _print(key, f"{(billing.get('usage') or {}).get(key, 0)} / {limit}")
    if billing.get("period_end"):
        _print("resets", time.strftime("%Y-%m-%d", time.gmtime(billing["period_end"])))


def main(args):
    client = Platform()
    verbs = {"finetune": _cmd_finetune, "generate": _cmd_generate, "jobs": _cmd_jobs,
             "models": _cmd_models, "files": _cmd_files, "billing": _cmd_billing}
    if not args.verb:
        raise SystemExit("needle platform finetune | generate | jobs | models | files | billing")
    try:
        verbs[args.verb](client, args)
    except PlatformError as error:
        raise SystemExit(f"platform error {error}")
