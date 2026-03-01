```
         ┌────────────────────────────────────────────────────────┐
         │                                                        │
         │       ┌─┐┌─┐┌─┐┌┬┐┬ ┬┌─┐  ┌┐┌┌─┐┌─┐┌┬┐┬  ┌─┐           │
         │       │  ├─┤│   │ │ │└─┐  │││├┤ ├┤  │││  ├┤            │
         │       └─┘┴ ┴└─┘ ┴ └─┘└─┘  ┘└┘└─┘└─┘─┴┘┴─┘└─┘           │
         │       ...the tiny model to rule them all...            │
         └────────────────────────────────────────────────────────┘

  Architecture                                        Training Pipeline
  ────────────                                        ─────────────────

                          ┌─────────────┐
                          │   Softmax   │             ┌──────────────────────┐
                          └──────┬──────┘             │ Forward pass with    │
                          ┌──────┴──────┐             │ INT4 fake-quantized  │
                          │  Linear (T) │  ← tied     │ weights (g=32, STE)  │
                          └──────┬──────┘             └──────────┬───────────┘
                          ┌──────┴──────┐                        │
                          │  ZCRMSNorm  │             ┌──────────┴───────────┐
                          └──────┬──────┘             │ Muon  (2D kernels)   │
                       ┌─────────┴─────────┐          │ AdamW (everything    │
                       │   Decoder x N     │          │       else)          │
                       │ ┌───────────────┐ │          │ WSD LR schedule      │
                       │ │ Masked Self   │ │          └──────────┬───────────┘
                       │ │ Attn + RoPE   │ │                     │
                       │ ├───────────────┤ │          ┌──────────┴───────────┐
  ┌──────────────┐  S  │ │   Cross       │ │          │ EMA params (β=0.999) │
  │ MemoryMixer  │─────────▶ Attention   │ │          └──────────┬───────────┘
  │ Encoder x N  │     │ ├───────────────┤ │                     │
  │              │     │ │ Feed-Forward  │ │          ┌──────────┴───────────┐
  │ ┌──────────┐ │     │ │   (dReLU)     │ │          │ Block Prune          │
  │ │Pack:     │ │     │ └───────────────┘ │          │  after epoch 1       │
  │ │ S←X Attn │ │     └─────────┬─────────┘          │  group magnitude     │
  │ ├──────────┤ │        ┌──────┴──────┐             │  lock sparsity mask  │
  │ │Mix:      │ │        │  Embedding  │  ← shared   └──────────┬───────────┘
  │ │ MLP-Mixer│ │        └──────┬──────┘                        │
  │ │ on S     │ │               │                    ┌──────────┴───────────┐
  │ ├──────────┤ │         ┌─────┴─────┐              │ Layer Prune          │
  │ │Local:    │ │         │  Decoder  │              │  after epoch 2       │
  │ │ FFN on X │ │         │   Input   │              │  L1 block scoring    │
  │ └──────────┘ │         └───────────┘              │  keep ≥1 enc & dec   │
  │  S ∈ (M, d)  │                                    └──────────┬───────────┘
  └──────┬───────┘                                               │
  ┌──────┴───────┐                                    ┌──────────┴───────────┐
  │  Embedding   │ ← shared                           │ Checkpoint           │
  └──────┬───────┘                                    │  sparse + INT4       │
         │                                            └──────────────────────┘
    ┌────┴────┐
    │ Encoder │
    │  Input  │
    └─────────┘

    d=128 · 4 heads · 2 KV heads · 64 memory slots
    dReLU · ZCRMSNorm · RoPE · INT4 QAT · Muon
```

## Usage

```
git clone https://github.com/cactus-compute/needle.git

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

## TPU Factsheet

```
┌────────────────────┬───────────────────┬──────────────────────┬────────────────────────────┐
│                    │       v5e         │        v5p           │      v6e (Trillium)        │
├────────────────────┼───────────────────┼──────────────────────┼────────────────────────────┤
│ Optimized for      │ Train + inference │ Training (max perf)  │ Train + inference          │
│ HBM per chip       │ 16 GB             │ 95 GB                │ 32 GB                      │
│ FLOPS (BF16)       │ 197 TFLOPS        │ 459 TFLOPS           │ 918 TFLOPS                 │
│ HBM bandwidth      │ 819 GB/s          │ 2,765 GB/s           │ 1,640 GB/s                 │
│ ICI bandwidth      │ 1,600 Gbps        │ 4,800 Gbps           │ 3,584 Gbps                 │
│ On-demand/chip/hr  │ $1.20             │ $4.20                │ $2.70                      │
│ Spot/chip/hr       │ $0.60             │ $2.10                │ $1.35                      │
│ Perf per $         │ 1x (baseline)     │ 0.5x                 │ 2x                         │
├────────────────────┼───────────────────┴──────────────────────┴────────────────────────────┤
│                    │                                                                       │
│ DATASET            │                                                                       │
│  Text              │ 100B tokens                                                           │
│  Audio             │ 200k × 20s = ~200M audio tokens + ~13M transcription tokens           │
│  Effective total   │ ~100.5B equivalent tokens (audio has ~2-3x encoder overhead)          │
│  Storage           │ ~5 GB audio (compressed) + ~400 GB text corpus                        │
│                    │                                                                       │
├────────────────────┼───────────────────┬──────────────────────┬─────────────┬──────────────┤
│ 300M multimodal    │   v5litepod-4     │      v6e-4           │    v6e-8    │    v6e-16    │
├────────────────────┼───────────────────┼──────────────────────┼─────────────┼──────────────┤
│ Chips              │ 4                 │ 4                    │ 8           │ 16           │
│ Total HBM          │ 64 GB             │ 128 GB               │ 256 GB      │ 512 GB       │
│ Est. time          │ ~16-21 days       │ ~4-5 days            │ ~2-3 days   │ ~1-1.5 days  │
│ Spot $/hr          │ $2.40             │ $5.40                │ $10.80      │ $21.60       │
│ Est. total cost    │ ~$900-1,200       │ ~$550-700            │ ~$550-750   │ ~$550-750    │
└────────────────────┴───────────────────┴──────────────────────┴─────────────┴──────────────┘
```

## Setup For TPU/GCP 

- Setup gcloud 1: download the `macOS ARM` from [here](https://docs.cloud.google.com/sdk/docs/install-sdk) and uzip.
- Setup gcloud 2: open terminal, cd to ypur downloads and run `./google-cloud-sdk/install.sh`
- Setup gcloud 3: run `gloud init`, sign in with cactus email, should prompt for project
- Setup gcloud 4: else, set the project with `gcloud config set project needle-488623`
- setup gcloud 5: run `gcloud help` and read carefully

## TPU Guide

```
needle tpu [command]

  ┌───────────────────────────────────────────────────────────────────┐
  │                                                                   │
  │   create NAME             Create TPU (auto-finds zone)            │
  │     --type STR            Accelerator type (default: v5litepod-4) │
  │     --version STR         TPU OS (default: tpu-ubuntu2204-base)   │
  │                                                                   │
  │   connect NAME            SSH config + first connect (auto-zone)  │
  │   claude NAME             Install Claude Code on instance         │
  │                                                                   │
  │   stop NAME               Stop instance (keeps disk)              │
  │   start NAME              Restart a stopped instance              │
  │   delete NAME             Delete instance (prompts confirmation)  │
  │   list                    List all TPU instances                  │
  │                                                                   │
  │     --zone ZONE           Override auto-detected zone (optional)  │
  │                                                                   │
  └───────────────────────────────────────────────────────────────────┘

  Quota increases:
   https://console.cloud.google.com/iam-admin/quotas?project=needle-488623
```

## Example Workflow

```
1. Create an instance (auto: finds zone → installs Claude Code → connects via SSH)
   needle tpu create my-experiment
   (exit with 'exit' or Ctrl+D)

2. Reconnect anytime (exit with 'exit' or Ctrl+D)
   ssh my-experiment
   or VS Code: click the '><' in the bottom left → select my-experiment

--- run from the instance ---

3. Clone the repo on your instance
   git clone https://github.com/cactus-compute/needle.git
   cd needle

4. Install needle (follow instruction to setup wandb)
   source ./setup

5. Use needle as you normally would locally, like training
   needle train --wandb

--- back on your Mac ---

6. Stop when done (saves disk, stops billing)
   needle tpu stop my-experiment

7. (Optional) Delete instance when no longer needed
   needle tpu delete my-experiment
```