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
                          └──────┬──────┘             │ Forward with MRL     │
                    ┌─────┬──────┴──────┬─────┐       │ loss at dims         │
                    │  @E[:d'] for each d'    │       │ {512,256,128,64}     │
                    │  in mrl_dims            │       │ + INT4 QAT (g=32)    │
                    └─────┬──────┬──────┬─────┘       └──────────┬───────────┘
                          ┌──────┴──────┐                        │
                          │  Linear (T) │  ← tied     ┌──────────┴───────────┐
                          └──────┬──────┘             │ Muon  (2D kernels)   │
                          ┌──────┴──────┐             │ AdamW (everything    │
                          │  ZCRMSNorm  │             │       else)          │
                          └──────┬──────┘             │ WSD LR schedule      │
                       ┌─────────┴─────────┐          └──────────┬───────────┘
                       │  Decoder x N_dec  │                     │
                       │ ┌───────────────┐ │          ┌──────────┴───────────┐
                       │ │ Masked Self   │ │          │ EMA params (β=0.999) │
                       │ │ Attn + RoPE   │ │          └──────────┬───────────┘
                       │ ├───────────────┤ │                     │
  ┌──────────────┐  S  │ │   Cross       │ │          ┌──────────┴───────────┐
  │ MemoryMixer  │─────────▶ Attention   │ │          │ Block Prune          │
  │ Encoder      │     │ ├───────────────┤ │          │  after epoch 1       │
  │  x N_enc     │     │ │ Feed-Forward  │ │          │  group magnitude     │
  │              │     │ │   (dReLU)     │ │          │  lock sparsity mask  │
  │ ┌──────────┐ │     │ └───────────────┘ │          └──────────┬───────────┘
  │ │Pack:     │ │     └─────────┬─────────┘                     │
  │ │ S←X Attn │ │        ┌──────┴──────┐             ┌──────────┴───────────┐
  │ │ RoPE keys│ │        │  Embedding  │  ← shared   │ MRL Checkpoint       │
  │ ├──────────┤ │        └──────┬──────┘             │  save per mrl_dim    │
  │ │Mix:      │ │               │                    │  {512,256,128,64}    │
  │ │ MLP-Mixer│ │         ┌─────┴─────┐              │  sparse + INT4       │
  │ │ on S     │ │         │  Decoder  │              └──────────────────────┘
  │ ├──────────┤ │         │   Input   │
  │ │Local:    │ │         └───────────┘
  │ │ FFN on X │ │
  │ └──────────┘ │
  │  S ∈ (M, d)  │
  └──────┬───────┘
  ┌──────┴───────┐
  │  Embedding   │ ← shared
  └──────┬───────┘
         │
    ┌────┴────┐
    │ Encoder │
    │  Input  │
    └─────────┘

    d=max(mrl_dims) · 4 heads · 2 KV heads · 64 memory slots
    SentencePiece BPE (8192) · dReLU · ZCRMSNorm · RoPE
    Matryoshka dims · INT4 QAT · Muon + AdamW
```

## Usage

```
git clone https://github.com/cactus-compute/needle.git

source ./setup

needle [command]

  ┌───────────────────────────────────────────────────────────────────┐
  │                                                                   │
  │   train                                                           │
  │     --toy                  Use toy config for quick iteration     │
  │     --epochs INT           Training epochs (default: 2)           │
  │     --batch-size INT       Batch size (default: 32)               │
  │     --lr FLOAT             AdamW learning rate (default: 3e-4)    │
  │     --muon-lr FLOAT        Muon learning rate (default: 0.02)     │
  │     --d-model INT          Model dim (default: max of mrl-dims)   │
  │     --num-heads INT        Attention heads (default: 4)           │
  │     --num-layers INT       Encoder layers (default: 12)           │
  │     --num-dec-layers INT   Decoder layers (default: 4)            │
  │     --max-enc-len INT      Max encoder seq length (default: 256)  │
  │     --max-dec-len INT      Max decoder seq length (default: 256)  │
  │     --max-samples INT      Training samples (default: 1000000)    │
  │     --mrl-dims INT [...]   MRL dim targets (default: 512 256 128) │
  │     --sparsity-ratio FLOAT Block prune ratio (default: 0.33)      │
  │     --layer-prune-ratio FL Layer prune ratio (default: 0.0)       │
  │     --group-size INT       Quant/prune group size (default: 32)   │
  │     --activation STR       drelu|swiglu|geglu (default: drelu)    │
  │     --warmup-ratio FLOAT   LR warmup ratio (default: 0.05)       │
  │     --eval-every INT       Val eval interval (default: 1000)      │
  │     --wandb                Enable W&B logging                     │
  │     --checkpoint PATH      Resume from checkpoint                 │
  │     --checkpoint-dir DIR   Checkpoint directory                   │
  │     --seed INT             Random seed (default: 42)              │
  │                                                                   │
  │   sweep                                                           │
  │     --sweep-config PATH    Sweep YAML config                      │
  │     --project STR          W&B project name (default: needle-v1)  │
  │     --count INT            Number of trials (default: 20)         │
  │                                                                   │
  │   run                                                             │
  │     --checkpoint PATH      Path to model checkpoint (required)    │
  │     --prompts STR [...]    One or more prompts to continue        │
  │     --max-len INT          Max tokens to generate (default: 128)  │
  │     --temperature FLOAT    Sampling temperature (default: 0.8)    │
  │     --seed INT             Random seed (default: 0)               │
  │                                                                   │
  │   test                                                            │
  │     --checkpoint PATH      Path to model checkpoint (required)    │
  │     --batch-size INT       Batch size (default: 32)               │
  │     --max-eval-samples INT Evaluation samples (default: 1000)     │
  │     --max-gen-len INT      Max generation length (default: 128)   │
  │     --temperature FLOAT    Sampling temperature (default: 0.8)    │
  │     --throughput-runs INT  Throughput runs (default: 10)          │
  │                                                                   │
  │   evaluate                                                        │
  │     --checkpoint PATH      Path to model checkpoint (required)    │
  │     --benchmarks [...]     wikitext2 lambada hellaswag arc_easy   │
  │     --max-samples INT      Samples per benchmark (default: 500)   │
  │                                                                   │
  │   tpu                                                             │
  │     create NAME            Create TPU (auto-finds zone)           │
  │       --type STR           Accelerator (default: v6e-8)           │
  │       --version STR        TPU OS (auto-detected from --type)     │
  │     connect NAME           SSH config + connect (auto-zone)       │
  │     claude NAME            Install Claude Code on instance        │
  │     stop NAME              Stop instance (auto-zone)              │
  │     start NAME             Start stopped instance (auto-zone)     │
  │     delete NAME            Delete instance (auto-zone)            │
  │     list                   List all TPU instances                 │
  │       --zone ZONE          Override auto-detected zone            │
  │                                                                   │
  └───────────────────────────────────────────────────────────────────┘
```

## TPU Factsheet

```
  Trillium (v6e)
  ──────────────────────────────────────────
  HBM per chip        32 GB
  BF16 FLOPS          918 TFLOPS
  HBM bandwidth       1,640 GB/s
  ICI bandwidth       3,584 Gbps
  On-demand/chip/hr   $2.70 (US regions)
  ──────────────────────────────────────────

  Dataset
  ──────────────────────────────────────────
  Text                100B tokens (~400 GB)
  Audio               Emilia subset
                      200k × 20s clips (~1.1k hrs)
                      ~200M audio tokens
                      ~13M transcription tokens
                      ~27 GB (from ~4.5 TB full dataset)
  ──────────────────────────────────────────

  300M multimodal training estimates
  ──────────────────────────────────────────
  Total FLOPs         6 × 300M × 100B = 1.8e20
  MFU (300M model)    ~15% → ~138 effective TFLOPS/chip
  Audio overhead      <1% of total (negligible)

  ┌────────────────────┬──────────┬──────────┐
  │                    │  v6e-8   │  v6e-16  │
  ├────────────────────┼──────────┼──────────┤
  │ Chips              │ 8        │ 16       │
  │ Total HBM          │ 256 GB   │ 512 GB   │
  │ Scaling eff.       │ 0.9×     │ 0.8×     │
  │ Eff. TFLOPS        │ 992      │ 1,766    │
  │ Est. time          │ ~50h     │ ~29h     │
  │ On-demand $/hr     │ $21.60   │ $43.20   │
  │ Est. total cost    │ ~$1,080  │ ~$1,250  │
  └────────────────────┴──────────┴──────────┘
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
  │     --type STR            Accelerator type (default: v6e-8)       │
  │     --version STR         TPU OS (auto-detected from --type)      │
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

3. Create a GitHub Personal Access Token (PAT)
   GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
   Generate a token with 'repo' scope

4. Clone the repo on your instance using your PAT
   git clone https://<your-PAT>@github.com/cactus-compute/needle.git
   cd needle

5. Install needle (follow instruction to setup wandb)
   source ./setup

6. Use needle as you normally would locally, like training
   needle train --wandb

7. Use tmux for long training runs (survives SSH disconnects)
   tmux new -s train          # start a named session
   needle train --wandb       # run training inside it
   Ctrl+B, then D             # detach (keeps running)
   tmux attach -t train       # reattach later

--- back on your Mac ---

8. Stop when done (saves disk, stops billing)
   needle tpu stop my-experiment

9. (Optional) Delete instance when no longer needed
   needle tpu delete my-experiment
```
