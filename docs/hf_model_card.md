---
license: mit
library_name: jax
tags:
  - function-calling
  - tool-use
  - encoder-decoder
  - edge
  - on-device
  - jax
  - flax
---

# Needle

We distilled Gemini 3.1 into a 26m parameter "[Simple Attention Network](docs/simple_attention_networks.md)" that you can even finetune locally on your Mac/PC.
In production, Needle runs on [Cactus](https://github.com/cactus-compute/cactus) at 6000 toks/sec prefill and 1200 decode speed. 
Weights are fully open on [Cactus-Compute/needle](https://huggingface.co/Cactus-Compute/needle), as well as the dataset generation. 

| | |
|---|---|
| Parameters | 26M |
| Architecture | Encoder-decoder, pure attention (no FFN) |
| Encoder | 12 layers, GQA (8H/4KV), RoPE, gated residuals |
| Decoder | 8 layers, self-attn + cross-attn, gated residuals |
| d_model | 512 |
| Vocab | 8192 (SentencePiece BPE) |
| Norm | ZCRMSNorm (zero-centered, init=0) |
| Precision | bfloat16 (INT4 QAT during training) |
| Pretraining | 200B tokens on 16x TPU v6e (27hrs) |
| Post-training | 2B tokens of function call data (45mins) |

```
d=512, 8H/4KV, BPE=8192
                                  ┌──────────────┐
                                  │  Tool Call   │
                                  └──────┬───────┘
                                        ┌┴──────────┐
                                        │  Softmax  │
                                        └─────┬─────┘
                                        ┌─────┴─────┐
                                        │ Linear (T)│  <- tied
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
  │ Encoder x 12 │─────────────────────>Cross Attn    ││
  │              │                   ││ Gated Residual││
  │ ┌──────────┐ │                   │└───────────────┘│
  │ │ZCRMSNorm │ │                   └────────┬────────┘
  │ │Self Attn │ │                      ┌─────┴─────┐
  │ │ GQA+RoPE │ │                      │ Embedding │  <- shared
  │ │Gated Res │ │                      └─────┬─────┘
  │ │          │ │                    ┌───────┴────────┐
  │ │ (no FFN) │ │                    │[EOS]<tool_call>│
  │ └──────────┘ │                    │ + answer       │
  │              │                    └────────────────┘
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

## Quickstart

```bash
git clone https://github.com/cactus-compute/needle.git
cd needle && source ./setup
needle playground
```

Opens a web UI at http://127.0.0.1:7860 where you can test and finetune on your own tools. Weights are auto-downloaded.

## Usage (Python)

```python
from needle import load_checkpoint, generate, SimpleAttentionNetwork, get_tokenizer

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

Finetune on your own tools via the web UI or CLI:

```bash
# Web UI (generates data via Gemini, trains, evaluates, bundles result)
needle playground

# CLI (auto-downloads weights if not local)
needle finetune data.jsonl
```

## Links

- [Needle](https://github.com/cactus-compute/needle) - training, finetuning, and inference code
- [Cactus](https://github.com/cactus-compute/cactus) - on-device runtime (6000 tok/s prefill, 1200 tok/s decode)
- [Simple Attention Networks](https://github.com/cactus-compute/needle/blob/main/docs/simple_attention_networks.md) - architecture details

## License

MIT

## Citation

```
@misc{ndubuaku2026needle,
  title={Needle},
  author={Henry Ndubuaku and Jakub Mroz and Karen Mosoyan and Roman Shemet and Parkirat Sandhu and Satyajit Kumar and Noah Cylich and Justin H. Lee},
  year={2026},
  url={https://github.com/cactus-compute/needle}
}
```
