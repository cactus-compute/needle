```
  ┌────────────────────────────────────────────────────────┐
  │                                                        │
  │      ┌─┐┌─┐┌─┐┌┬┐┬ ┬┌─┐  ┌┐┌┌─┐┌─┐┌┬┐┬  ┌─┐            │
  │      │  ├─┤│   │ │ │└─┐  │││├┤ ├┤  │││  ├┤             │
  │      └─┘┴ ┴└─┘ ┴ └─┘└─┘  ┘└┘└─┘└─┘─┴┘┴─┘└─┘            │
  │      ...the tiny model to rule them all...             │
  └────────────────────────────────────────────────────────┘

                          ┌─────────────┐
                          │   Softmax   │
                          └──────┬──────┘
                          ┌──────┴──────┐
                          │  Linear (T) │  ← tied weights
                          └──────┬──────┘
                          ┌──────┴──────┐
                          │  LayerNorm  │
                          └──────┬──────┘
                       ┌─────────┴─────────┐
                       │   Decoder x 2     │
                       │ ┌───────────────┐ │
                       │ │ Masked Self   │ │
  ┌──────────────┐     │ │ Attn + RoPE   │ │
  │              │     │ ├───────────────┤ │
  │ Encoder x 2  │     │ │   Cross       │ │
  │ ┌──────────┐ │────────▶  Attention   │ │
  │ │  Self    │ │     │ ├───────────────┤ │
  │ │Attn+RoPE │ │     │ │ Feed-Forward  │ │
  │ ├──────────┤ │     │ └───────────────┘ │
  │ │  Feed-   │ │     └─────────┬─────────┘
  │ │ Forward  │ │        ┌──────┴──────┐
  │ └──────────┘ │        │  Embedding  │  ← shared
  └──────┬───────┘        └──────┬──────┘
  ┌──────┴───────┐               │
  │  Embedding   │ ← shared      │
  └──────┬───────┘               │
         │                       │
    ┌────┴────┐            ┌─────┴─────┐
    │ Encoder │            │  Decoder  │
    │  Input  │            │   Input   │
    └─────────┘            └───────────┘

    ~7.5M params · d=128 · 4 heads · 2+2 layers · RoPE
```

## Usage

```
git clone https://github.com/cactus-compute/model.git

source ./setup

needle [command]

  ┌───────────────────────────────────────────────────────────────────┐
  │                                                                   │
  │   train                                                           │
  │     --epochs INT            Training epochs (default: 1)          │
  │     --batch-size INT        Batch size (default: 32)              │
  │     --lr FLOAT              AdamW learning rate (default: 3e-4)   │
  │     --muon-lr FLOAT         Muon learning rate (default: 0.02)    │
  │     --d-model INT           Model dimension (default: 128)        │
  │     --num-heads INT         Attention heads (default: 4)          │
  │     --num-layers INT        Encoder/decoder layers (default: 2)   │
  │     --dropout FLOAT         Dropout rate (default: 0.1)           │
  │     --max-enc-len INT       Max encoder seq length (default: 128) │
  │     --max-dec-len INT       Max decoder seq length (default: 128) │
  │     --max-samples INT       Training samples (default: 20000)     │
  │     --warmup-ratio FLOAT    LR warmup ratio (default: 0.05)       │
  │     --wandb                 Enable W&B logging                    │
  │     --checkpoint-dir DIR    Checkpoint directory                  │
  │     --seed INT              Random seed (default: 42)             │
  │                                                                   │
  │   sweep                                                           │
  │     --sweep-config PATH     Sweep YAML config                     │
  │     --project STR           W&B project name (default: needle-v1) │
  │     --count INT             Number of trials (default: 20)        │
  │                                                                   │
  │   run                                                             │
  │     --checkpoint PATH       Path to model checkpoint (required)   │
  │     --prompts STR [...]     One or more prompts to continue       │
  │     --max-len INT           Max tokens to generate (default: 128) │
  │     --temperature FLOAT     Sampling temperature (default: 0.8)   │
  │     --seed INT              Random seed (default: 0)              │
  │                                                                   │
  │   test                                                            │
  │     --checkpoint PATH       Path to model checkpoint (required)   │
  │     --batch-size INT        Batch size (default: 32)              │
  │     --max-eval-samples INT  Evaluation samples (default: 1000)    │
  │     --max-gen-len INT       Max generation length (default: 128)  │
  │     --temperature FLOAT     Sampling temperature (default: 0.8)   │
  │     --throughput-runs INT   Throughput runs (default: 10)         │
  │                                                                   │
  │   evaluate                                                        │
  │     --checkpoint PATH       Path to model checkpoint (required)   │
  │     --benchmarks [...]      wikitext2 lambada hellaswag arc_easy  │
  │     --max-samples INT       Samples per benchmark (default: 500)  │
  │                                                                   │
  └───────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
  src/
   ├── model.py ······ Transformer architecture
   ├── data.py ······· TinyStories loading & preprocessing
   ├── train.py ······ Training loop
   ├── run.py ········ Story generation from prompts
   ├── test.py ······· Throughput & quality benchmarks
   ├── evaluate.py ··· NLP benchmark evaluation
   └── cli.py ········ CLI entry point
```
