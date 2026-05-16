# Needle

<img src="assets/banner.png" alt="Logo" style="border-radius: 30px; width: 100%;">

We distilled Gemini 3.1 into a 26m parameter "[Simple Attention Network](docs/simple_attention_networks.md)" that you can even finetune locally on your Mac/PC.
In production, Needle runs on [Cactus](https://github.com/cactus-compute/cactus) at 6000 toks/sec prefill and 1200 decode speed. 
Weights are fully open on [Cactus-Compute/needle](https://huggingface.co/Cactus-Compute/needle), as well as the dataset generation. 

```
d=512, 8H/4KV, BPE=8192
                                  ┌──────────────┐
                                  │  Tool Call   │
                                  └──────┬───────┘
                                        ┌┴──────────┐
                                        │  Softmax  │
                                        └─────┬─────┘
                                        ┌─────┴─────┐
                                        │ Linear (T)│  ← tied
                                        └─────┬─────┘
                                        ┌─────┴─────┐
                                        │ ZCRMSNorm │
                                        └─────┬─────┘
                                     ┌────────┴────────┐
                                     │ Decoder x 8     │
                                     │┌───────────────┐│
                                     ││ ZCRMSNorm     ││
                                     ││ Masked Self   ││
                                     ││ Attn + RoPE   ││
                                     ││ Gated Residual││
                                     │├───────────────┤│
  ┌──────────────┐                   ││ ZCRMSNorm     ││
  │ Encoder x 12 │──────────────────────▶Cross Attn   ││
  │              │                   ││ Gated Residual││
  │ ┌──────────┐ │                   │└───────────────┘│
  │ │ZCRMSNorm │ │                   └────────┬────────┘
  │ │Self Attn │ │                      ┌─────┴─────┐
  │ │ GQA+RoPE │ │                      │ Embedding │  ← shared
  │ │Gated Res │ │                      └─────┬─────┘
  │ │          │ │                    ┌───────┴───────-┐
  │ │ (no FFN) │ │                    │[EOS]<tool_call>│
  │ └──────────┘ │                    │ + answer       │
  │              │                    └───────────────-┘
  └──────┬───────┘
         │
    ┌────┴──────┐
    │ Embedding │
    └────┬──────┘
         │
    ┌────┴──────┐
    │   Text    │
    │  query    │
    └───────────┘
```

- Pretrained on 16 TPU v6e for 200B tokens (27hrs). 
- Post-trained on 2B tokens of single-shot function call dataset (45mins). 

Needle is an experimental run for Simple Attention Networks, geared at redefining tiny AI for consumer devices (phones, watches, glasses...).
So while it beats FunctionGemma-270m, Qwen-0.6B, Graninte-350m, LFM2.5-350m on single-shot function call for personal AI,
Those model are have more scope/capacity and excel in conversational settings. Also, small models can be finicky. 
Please use the UI in the next section to test on your own tools, and finetune accordingly, at the click of a button. 

## Quickstart

```bash
git clone https://github.com/cactus-compute/needle.git
cd needle && source ./setup
needle playground
```

Opens a web UI at http://127.0.0.1:7860 where you can test and finetune on your own tools. Weights are auto-downloaded.

## Usage (Python)

```python
from needle import SimpleAttentionNetwork, load_checkpoint, generate, get_tokenizer

params, config = load_checkpoint("checkpoints/needle.pkl")
model = SimpleAttentionNetwork(config)
tokenizer = get_tokenizer()

result = generate(
    model, params, tokenizer,
    query="What's the weather in San Francisco?",
    tools='[{"name":"get_weather","description":"Get current weather for a city.","parameters":{"location":{"type":"string","description":"City name.","required":true}}}]',
    stream=False,
)
print(result)
# [{"name":"get_weather","arguments":{"location":"San Francisco"}}]
```

## Finetuning

```bash
# Playground (generates data via Gemini, trains, evaluates, bundles result)
needle playground

# CLI (auto-downloads weights if not local)
needle finetune data.jsonl
```

### Data format

Each line in the JSONL file has three fields: `query`, `tools`, and `answers`.

**Tool schema:**
```json
{
  "name": "get_weather",
  "description": "Get current weather for a city.",
  "parameters": {
    "location": { "type": "string", "description": "City name.", "required": true }
  }
}
```

**Answer schema:**
```json
{ "name": "get_weather", "arguments": { "location": "Paris" } }
```

**Full JSONL example** (each line is one training example, `tools` and `answers` are JSON-encoded strings):
```jsonl
{"query": "What's the weather in Paris?", "tools": "[{\"name\":\"get_weather\",\"description\":\"Get current weather for a city.\",\"parameters\":{\"location\":{\"type\":\"string\",\"description\":\"City name.\",\"required\":true}}}]", "answers": "[{\"name\":\"get_weather\",\"arguments\":{\"location\":\"Paris\"}}]"}
{"query": "Turn off the lights", "tools": "[{\"name\":\"get_weather\",\"description\":\"Get current weather for a city.\",\"parameters\":{\"location\":{\"type\":\"string\",\"description\":\"City name.\",\"required\":true}}},{\"name\":\"toggle_lights\",\"description\":\"Toggle smart lights on or off.\",\"parameters\":{\"state\":{\"type\":\"string\",\"description\":\"on or off.\",\"required\":true}}}]", "answers": "[{\"name\":\"toggle_lights\",\"arguments\":{\"state\":\"off\"}}]"}
```

Provide at least **120 examples per tool** (100 train / 10 val / 10 test). Fewer examples will overfit — you'll see perfect training metrics but the model won't generalize. Vary query phrasing and include examples with multiple tools available.

### Using a finetuned model

Finetuning saves the best checkpoint as `checkpoints/needle_finetuned_<id>_best.pkl`:

```bash
needle run --checkpoint checkpoints/needle_finetuned_*_best.pkl \
  --query "What's the weather?" --tools '[{"name":"get_weather","description":"Get current weather for a city.","parameters":{"location":{"type":"string","description":"City name.","required":true}}}]'
```

```python
params, config = load_checkpoint("checkpoints/needle_finetuned_<id>_best.pkl")
model = SimpleAttentionNetwork(config)
result = generate(model, params, get_tokenizer(), query="...", tools='[...]', stream=False)
```

## CLI

```
needle playground                  Test and finetune via web UI
needle finetune <data.jsonl>       Finetune on your own data
needle run --query "..." --tools   Single inference
needle mcp-server                  Run Needle as a local MCP tool router
needle train                       Full training run
needle pretrain                    Pretrain on PleIAs/SYNTH
needle eval --checkpoint <path>    Evaluate a checkpoint
needle tokenize                    Tokenize dataset
needle generate-data               Synthesize training data via Gemini
needle tpu <action>                TPU management (see docs/tpu.md)
```

## Local MCP Server

Needle can be run as a local MCP tool router for agent workflows, with either stdio transport or a lightweight HTTP transport.

### Start the local MCP server

```bash
needle mcp-server --checkpoint checkpoints/needle.pkl
```

### Run over HTTP

```bash
needle mcp-server --checkpoint checkpoints/needle.pkl --transport http --host 127.0.0.1 --port 8765
```

### Claude Desktop / MCP orchestrator configuration

Use the local Needle process as a tool router for any MCP-compatible orchestrator.

```json
{
  "mcpServers": {
    "needle": {
      "command": "needle",
      "args": [
        "mcp-server",
        "--checkpoint",
        "/path/to/needle/checkpoints/needle.pkl",
        "--transport",
        "http",
        "--host",
        "127.0.0.1",
        "--port",
        "8765"
      ]
    }
  }
}
```

With `--transport http`, the server listens on `http://127.0.0.1:8765` and accepts JSON-RPC 2.0 requests at the root path.

### Health check

```bash
curl http://127.0.0.1:8765/health
```

### How the router works

1. The orchestrator sends a natural language query and the available tool metadata.
2. Needle returns a `tool_call`, `confidence`, and `match` flag.
3. The orchestrator can execute the chosen tool locally if confidence is high, or escalate to a cloud LLM if not.

```
@misc{ndubuaku2026needle,
  title={Needle},
  author={Henry Ndubuaku, Jakub Mroz,  Karen Mosoyan, Roman Shemet, Parkirat Sandhu, Satyajit Kumar, Noah Cylich, Justin H. Lee},
  year={2026},
  url={https://github.com/cactus-compute/needle}
}
```
