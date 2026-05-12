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

Needle is an experimental run for Simple Attention Networks, geared at redefining tiny AI for consumer devies (phones, watches, glasses...).
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
    tools='[{"name":"get_weather","parameters":{"location":"string"}}]',
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

## CLI

```
needle playground                  Test and finetune via web UI
needle finetune <data.jsonl>       Finetune on your own data
needle run --query "..." --tools   Single inference
needle train                       Full training run
needle pretrain                    Pretrain on PleIAs/SYNTH
needle eval --checkpoint <path>    Evaluate a checkpoint
needle tokenize                    Tokenize dataset
needle generate-data               Synthesize training data via Gemini
needle tpu <action>                TPU management (see docs/tpu.md)
```

```
@misc{ndubuaku2026needle,
  title={Needle},
  author={Henry Ndubuaku and Jakub Mroz and Karen Mosoyan and Roman Shemet and Parkirat Sandhu and Satyajit Kumar and Noah Cylich and Justin H. Lee},
  year={2026},
  url={https://github.com/cactus-compute/needle}
}
```