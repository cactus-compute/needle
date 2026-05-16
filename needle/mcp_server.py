"""Needle MCP server — exposes Needle as a local tool-dispatch MCP tool.

Any MCP-compatible orchestrator (Claude Desktop, custom agents) can use
Needle as a fast local router: Needle picks the tool call, the orchestrator
executes it.  If Needle's confidence is below the caller's threshold the
response signals a no-match so the orchestrator can escalate to a cloud LLM.

Usage:
    needle mcp-server --checkpoint checkpoints/needle.pkl

Claude Desktop config (~/.claude/claude_desktop_config.json):
    {
      "mcpServers": {
        "needle": {
          "command": "needle",
          "args": ["mcp-server", "--checkpoint", "/path/to/needle.pkl"]
        }
      }
    }
"""

import argparse
import json
import logging
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

logger = logging.getLogger(__name__)

_DISPATCH_TOOL = {
    "name": "dispatch",
    "description": (
        "Route a natural language query to the correct tool call using the "
        "Needle on-device model. Returns a tool call JSON and a confidence "
        "score. Use threshold to decide whether to trust the result or "
        "escalate to a cloud LLM."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural language user query to route.",
            },
            "tools": {
                "type": "array",
                "description": "Available tool definitions (name, description, parameters).",
                "items": {"type": "object"},
            },
            "threshold": {
                "type": "number",
                "description": (
                    "Confidence threshold in [0, 1]. When confidence is below "
                    "this value the response sets match=false. Default 0 "
                    "(always return best guess)."
                ),
                "default": 0.0,
            },
        },
        "required": ["query", "tools"],
    },
}


def _load_model(checkpoint_path: str):
    from .dataset.dataset import get_tokenizer
    from .model.architecture import SimpleAttentionNetwork
    from .model.run import load_checkpoint

    params, config = load_checkpoint(checkpoint_path)
    model = SimpleAttentionNetwork(config)
    tokenizer = get_tokenizer()
    return model, params, tokenizer


def _dispatch(model, params, tokenizer, query: str, tools: list, threshold: float):
    from .model.run import generate

    tools_json = json.dumps(tools, separators=(",", ":"))
    result, confidence = generate(
        model,
        params,
        tokenizer,
        query=query,
        tools=tools_json,
        stream=False,
        threshold=threshold,
        return_confidence=True,
    )
    try:
        tool_call = json.loads(result)
    except (json.JSONDecodeError, TypeError):
        tool_call = result

    return {
        "tool_call": tool_call,
        "confidence": round(confidence, 4),
        "match": not (isinstance(tool_call, dict) and tool_call.get("match") is False),
    }


def _handle_mcp_request(req: dict, model: Any, params: Any, tokenizer: Any):
    method = req.get("method", "")
    req_id = req.get("id")

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "serverInfo": {"name": "needle", "version": "1.0.0"},
                "capabilities": {"tools": {}},
            },
        }

    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"tools": [_DISPATCH_TOOL]},
        }

    if method == "tools/call":
        params = req.get("params", {})
        name = params.get("name")
        args = params.get("arguments", {})
        if name != "dispatch":
            return _error(req_id, -32601, f"Unknown tool: {name}")
        try:
            result = _dispatch(
                model,
                params,
                tokenizer,
                query=args["query"],
                tools=args.get("tools", []),
                threshold=float(args.get("threshold", 0.0)),
            )
        except Exception as exc:
            logger.exception("dispatch failed")
            return _error(req_id, -32603, str(exc))
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "content": [{"type": "text", "text": json.dumps(result)}],
                "isError": False,
            },
        }

    if method == "notifications/initialized":
        return None

    return _error(req_id, -32601, f"Method not found: {method}")


class _StdioMCPServer:
    """Minimal stdio MCP server (JSON-RPC 2.0, MCP 2024-11 spec)."""

    def __init__(self, model, params, tokenizer):
        self._model = model
        self._params = params
        self._tokenizer = tokenizer

    def run(self):
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
            except json.JSONDecodeError:
                continue
            resp = _handle_mcp_request(req, self._model, self._params, self._tokenizer)
            if resp is not None:
                sys.stdout.write(json.dumps(resp) + "\n")
                sys.stdout.flush()


class _HTTPMCPHandler(BaseHTTPRequestHandler):
    server_version = "NeedleMCP/1.0"
    protocol_version = "HTTP/1.1"

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0) or 0)
        body = self.rfile.read(content_length)
        try:
            req = json.loads(body.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Invalid JSON"}).encode("utf-8"))
            return

        resp = self.server.handle_mcp(req)
        if resp is None:
            resp = {}
        encoded = json.dumps(resp).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok"}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        return


class _HTTPMCPServer(HTTPServer):
    def __init__(self, server_address, RequestHandlerClass, model, params, tokenizer):
        super().__init__(server_address, RequestHandlerClass)
        self.model = model
        self.params = params
        self.tokenizer = tokenizer

    def handle_mcp(self, req: dict):
        return _handle_mcp_request(req, self.model, self.params, self.tokenizer)


def _run_http(model, params, tokenizer, host: str, port: int):
    server = _HTTPMCPServer((host, port), _HTTPMCPHandler, model, params, tokenizer)
    logger.warning("Needle MCP HTTP server listening on %s:%s", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


def main():
    parser = argparse.ArgumentParser(description="Needle MCP server")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/needle.pkl",
        help="Path to Needle checkpoint (.pkl)",
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="stdio",
        help="Server transport protocol for MCP (stdio or http)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind when using HTTP transport",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port to bind when using HTTP transport",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    print(f"[needle-mcp] Loading checkpoint: {args.checkpoint}", file=sys.stderr)
    model, params, tokenizer = _load_model(args.checkpoint)
    print("[needle-mcp] Ready", file=sys.stderr)

    if args.transport == "http":
        print(f"[needle-mcp] Listening on http://{args.host}:{args.port}", file=sys.stderr)
        _run_http(model, params, tokenizer, args.host, args.port)
    else:
        _StdioMCPServer(model, params, tokenizer).run()


if __name__ == "__main__":
    main()
