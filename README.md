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

               ┌──────────────┐  ┌──────────────┐
               │ Transcription│  │  Tool JSON   │   ┌──────────────────────┐
               └──────┬───────┘  └──────┬───────┘   │ MRL loss at dims     │
                      └───────┬─────────┘            │ + INT4 QAT (g=32)    │
                          ┌───┴───────┐              │ + z-loss + slot div  │
                          │  Softmax  │              │ mixed text/speech    │
                          └─────┬─────┘              │ batches              │
                    ┌─────┬─────┴─────┬─────┐        └──────────┬───────────┘
                    │  @E[:d'] for each d'  │                    │
                    │  in mrl_dims          │        ┌──────────┴───────────┐
                    └─────┬─────┬─────┬─────┘        │ Grad clip (norm 1.0) │
                          ┌─────┴─────┐              │ Muon  (2D kernels)   │
                          │ Linear (T)│  ← tied      │ AdamW (all else)     │
                          └─────┬─────┘              │ WSD LR schedule      │
                          ┌─────┴─────┐              └──────────┬───────────┘
                          │ ZCRMSNorm │                         │
                          └─────┬─────┘              ┌──────────┴───────────┐
                       ┌────────┴────────┐           │ EMA params (β=0.999) │
                       │ Decoder x N_dec │           └──────────┬───────────┘
                       │┌───────────────┐│                      │
                       ││ Masked Self   ││           ┌──────────┴───────────┐
                       ││ Attn + RoPE   ││           │   Sparsification     │
                       │├───────────────┤│           │  cubic schedule      │
  ┌──────────────┐  S  ││   Cross       ││           │  mask every N steps  │
  │ MemoryMixer  │────────▶ Attention   ││           └──────────┬───────────┘
  │ Encoder      │     │├───────────────┤│                      │
  │  x N_enc     │     ││  Gated FFN    ││           ┌──────────┴───────────┐
  │              │     │└───────────────┘│           │ Checkpoint           │
  │ ┌──────────┐ │     └────────┬────────┘           │  full params, can    │
  │ │Pack:     │ │        ┌─────┴─────┐              │  export MRL slices   │
  │ │ S←X Attn │ │        │ Embedding │  ← shared    └──────────────────────┘
  │ │ RoPE keys│ │        └─────┬─────┘
  │ ├──────────┤ │      ┌───────┴───────┐
  │ │Mix:      │ │      │ <transcribe>  │
  │ │ MLP-Mixer│ │      │ or <tools>    │
  │ │ on S     │ │      │ + target      │
  │ ├──────────┤ │      └───────────────┘
  │ │Local:    │ │
  │ │Gated FFN │ │
  │ │ on X     │ │
  │ └──────────┘ │
  │              │
  │  Slot Init   │  learnable + input-dependent
  │  DW Conv ↓2  │  stride-2 depthwise-separable
  │  S ∈ (M, d)  │
  └──────┬───────┘
  ┌──────┴───────┐
  │  Embedding   │ ← shared
  └──────┬───────┘
    ┌────┴─────────┐
    │ Text/Speech  │
    │    Input     │
    └──────────────┘

    d=512 (default) / 1536 (--full) · GQA · QK-norm
    SentencePiece BPE (8192) · gated dReLU · ZCRMSNorm · RoPE
    strided DW conv · MRL dims · INT4 QAT · Muon + AdamW
    text + speech encoder · <transcribe> / <tools> task routing
```

## Usage

```
git clone https://github.com/cactus-compute/needle.git

source ./setup

needle [command]

  ┌───────────────────────────────────────────────────────────────────┐
  │                                                                   │
  │   train                                                           │
  │     --full                  Use full 1B config (~1.17B params)    │
  │     --epochs INT            Training epochs (default: 1)          │
  │     --batch-size INT        Batch size (default: 32)              │
  │     --lr FLOAT              AdamW learning rate (default: 3e-4)   │
  │     --muon-lr FLOAT         Muon learning rate (default: 0.02)    │
  │     --d-model INT           Model dim (default: 512)              │
  │     --num-heads INT         Attention heads (default: 8)          │
  │     --num-kv-heads INT      KV heads for GQA (default: num-heads) │
  │     --num-layers INT        Encoder layers (default: 8)           │
  │     --num-dec-layers INT    Decoder layers (default: 4)           │
  │     --max-enc-len INT       Max encoder seq length (default: 256) │
  │     --max-dec-len INT       Max decoder seq length (default: 256) │
  │     --max-samples INT       Training samples (default: all)       │
  │     --mrl-dims INT [...]    MRL dim targets (default: 256 128 64) │
  │     --sparsity-ratio FLOAT  Block prune ratio (default: 0.5)     │
  │     --group-size INT        Quant/prune group size (default: 32)  │
  │     --prune-interval INT    Steps between mask updates (def: 100) │
  │     --prune-start-frac FL   Start pruning at frac (def: 0.33)    │
  │     --prune-end-frac FL     Lock mask at this frac (def: 0.67)   │
  │     --activation STR        drelu|swiglu|geglu (default: drelu)   │
  │     --warmup-ratio FLOAT    LR warmup ratio (default: 0.05)      │
  │     --eval-every INT        Val eval interval (default: 1000)     │
  │     --wandb                 Enable W&B logging                    │
  │     --checkpoint PATH       Resume from checkpoint                │
  │     --checkpoint-dir DIR    Checkpoint directory                  │
  │     --seed INT              Random seed (default: 42)             │
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

  100M quick-experiment estimates (--default)
  ──────────────────────────────────────────
  Total tokens        1B text + 1k hrs audio ≈ 1.2B
  Total FLOPs         6 × 102M × 1.2B ≈ 7.2e17
  MFU (100M model)    ~15% → ~138 effective TFLOPS/chip

  ┌────────────────────┬──────────┬──────────┬──────────┐
  │                    │  v6e-8   │  v6e-16  │  v6e-32  │
  ├────────────────────┼──────────┼──────────┼──────────┤
  │ Chips              │ 8        │ 16       │ 32       │
  │ Total HBM          │ 256 GB   │ 512 GB   │ 1024 GB  │
  │ Scaling eff.       │ 0.9×     │ 0.8×     │ 0.7×     │
  │ Eff. TFLOPS        │ 994      │ 1,766    │ 3,091    │
  │ Est. time          │ ~12min   │ ~7min    │ ~4min    │
  │ On-demand $/hr     │ $21.60   │ $43.20   │ $86.40   │
  │ Est. total cost    │ ~$4      │ ~$5      │ ~$6      │
  └────────────────────┴──────────┴──────────┴──────────┘

  1B multimodal training estimates (--full)
  ──────────────────────────────────────────
  Total tokens        100B text + 250k hrs audio ≈ 145B
  Total FLOPs         6 × 1.17B × 145B ≈ 1.0e21
  MFU (1B model)      ~30% → ~275 effective TFLOPS/chip

  ┌────────────────────┬──────────┬──────────┬──────────┐
  │                    │  v6e-8   │  v6e-16  │  v6e-32  │
  ├────────────────────┼──────────┼──────────┼──────────┤
  │ Chips              │ 8        │ 16       │ 32       │
  │ Total HBM          │ 256 GB   │ 512 GB   │ 1024 GB  │
  │ Scaling eff.       │ 0.9×     │ 0.8×     │ 0.7×     │
  │ Eff. TFLOPS        │ 1,980    │ 3,520    │ 6,160    │
  │ Est. time          │ ~143h    │ ~80h     │ ~46h     │
  │ On-demand $/hr     │ $21.60   │ $43.20   │ $86.40   │
  │ Est. total cost    │ ~$3,090  │ ~$3,460  │ ~$3,970  │
  └────────────────────┴──────────┴──────────┴──────────┘
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
