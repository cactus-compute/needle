# Needle

<img src="assets/banner.png" alt="Logo" style="border-radius: 30px; width: 70%;">

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
from needle.model.run import load_checkpoint, generate
from needle.model.architecture import EncoderDecoderTransformer
from needle.dataset.dataset import get_tokenizer

params, config = load_checkpoint("checkpoints/needle.pkl")
model = EncoderDecoderTransformer(config)
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


## Needle CLI

```
  ┌───────────────────────────────────────────────────────────────────┐
  │                                                                   │
  │   playground                   Web UI for inference + finetuning  │
  │     --checkpoint PATH          Model checkpoint (optional,        │
  │                                auto-downloads from HuggingFace)   │
  │     --port INT                 Server port (default: 7860)        │
  │     --host STR                 Bind address (default: 127.0.0.1)  │
  │                                                                   │
  │   run                                                             │
  │     --checkpoint PATH          Path to model checkpoint (required)│
  │     --query STR                Query text for tool-call generation│
  │     --tools STR                Tools JSON for tool-call generation│
  │     --max-len INT              Max tokens to generate (default:512│
  │     --seed INT                 Random seed (default: 0)           │
  │     --no-constrained           Disable constrained decoding       │
  │                                                                   │
  │   train                                                           │
  │     --name STR                 Experiment name (default: baseline)│
  │     --checkpoint PATH          Resume from checkpoint             │
  │     --init-from PATH           Init params from pretrained base   │
  │     --epochs INT               Training epochs (default: 1)       │
  │     --batch-size INT           Batch size (default: 64)           │
  │     --lr FLOAT                 AdamW learning rate (default: 3e-5)│
  │     --muon-lr FLOAT            Muon learning rate (default: 0.02) │
  │     --d-model INT              Model dim (default: 512)           │
  │     --num-heads INT            Attention heads (default: 8)       │
  │     --num-kv-heads INT         KV heads for GQA (default: 4)      │
  │     --num-layers INT           Encoder layers (default: 12)       │
  │     --num-dec-layers INT       Decoder layers (default: 8)        │
  │     --max-enc-len INT          Max encoder seq len (default: 1024)│
  │     --max-dec-len INT          Max decoder seq len (default: 512) │
  │     --max-samples INT          Training samples (default: all)    │
  │     --warmup-ratio FLOAT       LR warmup ratio (default: 0.05)    │
  │     --decay-ratio FLOAT        LR cosine decay ratio (default:0.05│
  │     --eval-every INT           Val eval interval (default: 1000)  │
  │     --max-eval-samples INT     Val samples limit (default: 5000)  │
  │     --contrastive-weight FL    CLIP loss weight (default: 0.1)    │
  │     --contrastive-dim INT      Projection dim (default: 128)      │
  │     --w-name FLOAT             Loss weight: tool names (def: 2.0) │
  │     --w-value FLOAT            Loss weight: arg values (def: 4.0) │
  │     --w-key FLOAT              Loss weight: arg keys (def: 1.5)   │
  │     --wandb                    Enable W&B logging                 │
  │     --checkpoint-dir DIR       Checkpoint directory               │
  │     --dtype STR                float32|bfloat16 (default: bfloat16│
  │     --seed INT                 Random seed (default: 42)          │
  │                                                                   │
  │   pretrain                                                        │
  │     --name STR                 Experiment name (default: pretrain)│
  │     --checkpoint PATH          Resume from checkpoint             │
  │     --resume-step INT          Override resume step               │
  │     --epochs INT               Training epochs (default: 1)       │
  │     --batch-size INT           Batch size (default: 128)          │
  │     --lr FLOAT                 AdamW learning rate (default: 3e-4)│
  │     --muon-lr FLOAT            Muon learning rate (default: 0.02) │
  │     --max-steps INT            Stop after N steps (default: full) │
  │     --save-every INT           Checkpoint interval (default: 1000)│
  │     --wandb                    Enable W&B logging                 │
  │     (also accepts --d-model, --num-heads, etc. same as train)     │
  │                                                                   │
  │   tokenize                                                        │
  │     --max-samples INT          Limit samples per split (dev/test) │
  │     --max-enc-len INT          Max encoder seq len (default: 1024)│
  │     --max-dec-len INT          Max decoder seq len (default: 512) │
  │     --shuffle-tools            Shuffle tool order (default: on)   │
  │     --no-shuffle-tools         Disable tool shuffling             │
  │     --max-tool-len INT         Max tool desc tokens (default: 256)│
  │                                                                   │
  │   eval                                                            │
  │     --checkpoint PATH          Path to model checkpoint (required)│
  │     --batch-size INT           Batch size (default: 32)           │
  │     --max-eval-samples INT     Evaluation samples (default: 5000) │
  │     --max-enc-len INT          Max encoder length (default: 1024) │
  │     --max-dec-len INT          Max decoder length (default: 512)  │
  │     --max-gen-len INT          Max generation length (default:512)│
  │     --throughput-runs INT      Throughput runs (default: 10)      │
  │     --tool-call-samples INT    Tool-call eval samples (default:200│
  │     --no-constrained           Disable constrained decoding       │
  │                                                                   │
  │   generate-data                                                   │
  │     --num-samples INT          Samples to generate (default: 500) │
  │     --batch-size INT           Examples per Gemini call (default:25│
  │     --workers INT              Parallel Gemini calls (default: 8) │
  │     --model STR                Gemini model override              │
  │     --dry-run                  Generate only, skip upload         │
  │     --output-jsonl PATH        Also save raw generations to JSONL │
  │     --upload-every INT         Merge+upload interval              │
  │                                                                   │
  │   tpu                          See docs/tpu.md                    │
  │                                                                   │
  └───────────────────────────────────────────────────────────────────┘
```

## TPU

This repo was used for training too, see [docs/tpu.md](docs/tpu.md) for GCP setup, TPU factsheet, and single/multi-host workflows.