# API

| Call | What it does |
| --- | --- |
| `needle.Needle(tools=None, system=None, weights=None, tool_index_path=None, buffer_size=65536)` | Create an agent bound to one toolset. `tools` takes decorated functions, Pydantic models, raw JSON schema dicts, or a JSON string. `system` carries environment facts. `weights` loads a tuned `.cact`. `tool_index_path` persists tool embeddings for large catalogues. |
| `agent.run(query, max_steps=8, max_new_tokens=256)` | Full agentic loop: the model picks calls, Needle executes your Python functions and feeds results back, and the final response carries the executed tool results as `results`. |
| `agent.complete(text, max_new_tokens=256)` | One turn. You execute the call and feed the result back yourself via the next `complete(...)`. |
| `agent.reset()` | Rewind the conversation, keep the tools loaded. |
| `needle.tool` | Decorator that turns a function into a tool schema (attached as `fn._needle_tool`). |
| `needle.Field(...)` | Per argument constraints, attached inline with `typing.Annotated` or passed as a default. |
| `needle.extract(text, schema, system=None, max_new_tokens=256, weights=None)` | One shot extraction. Returns a Pydantic instance if `schema` is a model, else a dict, or `None` if nothing matched. `weights` selects a tuned `.cact` and defaults to whatever the engine already has loaded. |

## Declaring tools

**Describe each argument and offer choices.** Needle reads a Google-style `Args:` block for per-parameter descriptions; a default makes an argument optional; a `Literal` becomes a fixed set the model must choose from (it cannot emit anything else).

```python
from typing import Literal

@needle.tool
def set_thermostat(temperature: int, mode: Literal["heat", "cool", "auto"] = "auto"):
    """Set the thermostat.

    Args:
        temperature: target temperature in Celsius
        mode: heating strategy to use
    """
    return {"temperature": temperature, "mode": mode}

agent = needle.Needle(tools=[set_thermostat])
agent.run("make it 21 and cool the room")
```

**Constrain the values** with `needle.Field`, attached inline via `Annotated`. Ranges, patterns, lengths, and item counts are compiled into the decode grammar, so the model can only ever emit values that satisfy them.

```python
from typing import Annotated

@needle.tool
def send_money(
    amount: Annotated[float, needle.Field(gt=0, le=10000, description="USD, up to 10,000")],
    to:     Annotated[str,   needle.Field(pattern=r"^@[a-z0-9_]+$", description="recipient handle")],
    memo:   Annotated[str,   needle.Field(max_length=80)] = "",
):
    "Send money to a handle."
    return {"sent": amount, "to": to}
```

`Field` supports `description`, `enum`, `const`, `ge`/`le`/`gt`/`lt`, `multiple_of`, `min_length`/`max_length`, `pattern`, `format`, `min_items`/`max_items`, and `unique_items`.

**By hand.** The decorator just builds a JSON schema; you can pass that schema directly, which is exactly what Needle consumes. This is how you set descriptions and constraints without the decorator:

```python
tools = [{
    "name": "set_lights",
    "description": "Turn a room's lights on or off and set brightness",
    "parameters": {
        "type": "object",
        "properties": {
            "room": {"type": "string", "description": "which room to control"},
            "on": {"type": "boolean"},
            "brightness": {"type": "integer", "minimum": 0, "maximum": 100},
        },
        "required": ["room", "on"],
    },
}]
agent = needle.Needle(tools=tools)
```

## Driving the loop

Prefer to drive the loop yourself instead of `run()`? `complete()` returns the raw call and you execute it:

```python
import json
response = agent.complete("dim the living room to 30")
if response["type"] == "call":
    result = set_lights(**response["function_calls"][0]["arguments"])
    response = agent.complete(json.dumps(result))   # feed the result back
```

Every turn returns one JSON object:

```json
{
  "type": "call",
  "success": true,
  "error": null,
  "error_code": null,
  "function_calls": [ { "name": "set_lights", "arguments": { "room": "living room", "on": true, "brightness": 30 } } ],
  "reasoning": "'living room' -> room; 'dim' -> on true, brightness 30",
  "confidence": 0.94,
  "prefill_tps": 4300.0,
  "decode_tps": 850.0,
  "peak_ram_mb": 28.5
}
```

## Behaviour

Needle solves every problem as a function call. The context declares what may be called; the model answers with calls. Performing an action and extracting structured data are the same operation, the only difference is what you declare.

- A request no declared tool can serve is refused with the empty call `[]`. That is the whole contract for off-topic input; there is no free-text fallback.
- Arguments contain only values evidenced by the input. An optional field with no evidence is omitted, not guessed; omission is the field-level `[]`.
- `reasoning` is the model's short derivation of each argument from its source span (`'ten minutes' -> minutes 10`). It is generated unconstrained; only the call itself is grammar-constrained, so the JSON cannot be malformed while the derivation stays legible.
- After you execute a call, pass the result back as the next `complete()`. The model continues from it, and later arguments may depend on earlier results: `search_for_contact` first, then `send_instant_message` with the returned `contact_id`. A final `"type": "respond"` with empty `function_calls` signals the loop is done; the answer is the tool results themselves, which `run()` collects on the final response as `results`. No free text is generated.
- A session shares one toolset. Later turns are bare queries against the same tools; `reset()` rewinds the conversation and keeps the tools loaded.

## Extraction

Extraction is not a separate mode - it is tool calling with one tool. Declare the record as the only schema and pass the content where the query goes; the returned call's `arguments` are the extracted fields. With one declared tool the grammar admits exactly one call of that name, so schema conformance is guaranteed rather than requested. Use the `extract()` helper for a typed result, or pass a plain schema and read the call:

```python
receipt = [{
    "name": "receipt",
    "description": "A purchase receipt shared as text",
    "parameters": {
        "type": "object",
        "properties": {
            "merchant": {"type": "string"},
            "total": {"type": "number"},
            "currency": {"type": "string"},
            "line_items": {"type": "array", "items": {"type": "object"}},
        },
        "required": ["merchant", "total"],
    },
}]
agent = needle.Needle(tools=receipt)
print(agent.complete("GreenMart receipt: oat milk 3.50, total 7.75 paid by visa")["function_calls"])
# -> [{'name': 'receipt', 'arguments': {'merchant': 'GreenMart', 'total': 7.75}}]
```

Because it is the same operation, everything else applies unchanged: `confidence` gates the extraction, unsupported input returns the empty call `[]`, and fine-tuning uses the same data format (the record as the tool, the passage as the query).

## System facts

An optional system turn carries environment state as facts, never instructions:

```python
agent = needle.Needle(tools=tools, system="date: 2026-07-21 Tue 14:30; locale: en-US; device: phone; battery: 62%")
```

Recognized keys are `date`, `locale`, `device`, `battery`, `network`, `location`, `user`, and `assistant`. The model resolves relative language against them: "tomorrow at 7" becomes an absolute time only when a `date:` fact licenses it, otherwise the human phrase passes through verbatim. `assistant:` declares the identity the model binds to. Needle trains with and without the turn, so omitting it is safe; instructions placed there do not steer the model.

## Tool retrieval

Five or fewer declared tools render directly. Above that, retrieval engages: at init every tool schema is embedded once by a built-in contrastive head, each turn embeds the query, and only the five highest-scoring tools enter the context, with the grammar rebuilt over just that subset. An unselected tool is unreachable, not merely unlikely. `tool_index_path` persists the embeddings on disk, keyed by a fingerprint over the schemas and the model; a matching fingerprint loads instantly, a changed schema re-embeds only what changed.

## Confidence

The `confidence` field is the minimum of two signals: a calibrated post-hoc head that scores the full prompt plus the call the model just produced, and the decoding probability of the call tokens. A call is accepted only when both agree, so the failure mode is escalation, not wrong execution. The contract: pick a threshold for your product, act at or above it, re-ask or route to a bigger model below it. Off-topic requests return the empty call `[]`.

Calibration holds for the base model only. Fine-tuning does not update the head, so an agent running tuned weights reports `confidence` as `None` and warns once at construction.

## Offline devices

The engine is fetched once and cached at `~/.cache/cactus-needle/<engine version>/`. Inference itself never touches the network, so an air gapped device only needs that one file in place. Three ways to get it there:

1. `needle fetch` downloads the engine for the current machine into the cache and prints the path. `--out <dir>` places it elsewhere. `--platform-tag manylinux2014_aarch64` fetches the build for a different device; tags follow the wheel names on the Hugging Face repo (`macosx_11_0_arm64`, `manylinux2014_x86_64`, `musllinux_1_2_aarch64`, `win_amd64`, `win_arm64`).
2. Copy the file to the same cache path on the device, or drop it inside the installed `needle/` package directory, which wins over the cache.
3. Set `NEEDLE_LIB_PATH=/path/to/libneedle.so` to load exactly that file and skip every lookup.

The Python package itself installs offline the standard way: `pip download cactus-needle` on a connected machine, then `pip install --no-index --find-links <dir> cactus-needle` on the device. On a device that must never attempt the network, also set `HF_HUB_OFFLINE=1` so a missing engine fails fast with a clear error instead of trying to download.
