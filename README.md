```
         ┌────────────────────────────────────────────────────────┐
         │                                                        │
         │       ┌─┐┌─┐┌─┐┌┬┐┬ ┬┌─┐  ┌┐┌┌─┐┌─┐┌┬┐┬  ┌─┐           │
         │       │  ├─┤│   │ │ │└─┐  │││├┤ ├┤  │││  ├┤            │
         │       └─┘┴ ┴└─┘ ┴ └─┘└─┘  ┘└┘└─┘└─┘─┴┘┴─┘└─┘           │
         │       ...the tiny model to rule them all...            │
         └────────────────────────────────────────────────────────┘

  Architecture
  ────────────

  Pure-attention encoder-decoder. No feedforward layers — each block is
  just gated self-attention (encoder) or gated self + cross attention
  (decoder). All mixing happens through attention.

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
                                     │ Decoder x 12    │
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
  │  DW Conv ↓2  │  stride-2 depthwise-separable
  │  + Pointwise │  (speech pathway only)
  └──────┬───────┘
         │
    ┌────┴──────┐    ┌──────────────┐
    │ Embedding │    │ Mel Proj     │
    │ (text)    │    │ (speech)     │
    └────┬──────┘    └──────┬───────┘
         │                  │
    ┌────┴──────┐    ┌──────┴───────┐
    │   Text    │    │   Audio      │
    │  query    │    │  waveform    │
    └───────────┘    └──────────────┘

  Heads
  ─────
  ┌─────────────────────────────────────────────────────────────┐
  │                                                             │
  │  Contrastive Head    CLIP-style tool retrieval              │
  │    encoder → mean_pool → Dense(d_model/4) → ReLU           │
  │    → Dense(contrastive_dim) → L2-normalize                  │
  │    learnable temperature (log_temp)                         │
  │                                                             │
  │  Confidence Head     cloud routing / hybrid inference       │
  │    encoder → mean_pool → Dense(d_model/4)                   │
  │    → ReLU → Dense(1) → sigmoid → [0, 1]                     │
  │    trained post-hoc on perplexity-derived labels            │
  │    high = handle locally, low = route to cloud              │
  │                                                             │
  └─────────────────────────────────────────────────────────────┘

  Training Pipeline
  ─────────────────
  ┌─────────────────────────────────────────────────────────────┐
  │  Contrastive loss + token-level weighting + z-loss          │
  │  Grad clip (norm 1.0) · Muon (2D kernels) · AdamW (rest)    │
  │  WSD LR schedule (warmup → hold → cosine decay)             │
  │  Grammar-constrained decoding (trie-based token masking)    │
  └─────────────────────────────────────────────────────────────┘

    d=512 · 12 enc / 12 dec layers · GQA (8H / 4KV) · QK-norm
    no FFN (pure attention) · gated residuals (init=0.5)
    SentencePiece BPE (8192) · ZCRMSNorm · RoPE
    text + speech encoder · <tool_call> task routing
    CLIP contrastive retrieval · confidence-based cloud routing
    token-level loss weighting · grammar-constrained decoding

  Data Pipeline (needle generate-data → needle tokenize → needle train)
  ─────────────────────────────────────────────────────────────────────

  ┌─────────────────────────────────────────────────────────────┐
  │  needle generate-data                                       │
  │                                                             │
  │  Gemini synthesis → Cactus-Compute/tool-calls (HuggingFace) │
  │  parallel workers · dedup · auto-merge · auto-upload        │
  │                                                             │
  │  Related:                                                   │
  │    needle merge-xlam        merge xlam-60k into dataset     │
  │    needle rebalance-tools   trim over-represented bins      │
  │    needle split-dataset     create train/val splits         │
  └─────────────────────────────────────────────────────────────┘
                         │
                         ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  needle tokenize                                            │
  │                                                             │
  │  GCS download (gs://cactus-dataset/tool_calls/)             │
  │       │                                                     │
  │       ▼                                                     │
  │  Unified dataset (.arrow)                                   │
  │       │                                                     │
  │       ▼                                                     │
  │  ┌──────────────┐                                           │
  │  │ SentencePiece│  trains tokenizer from corpus             │
  │  │ BPE tokenize │  special: <tool_call> <tools>             │
  │  │ (8192 vocab) │  byte_fallback, identity normalization    │
  │  └──────┬───────┘                                           │
  │         │                                                   │
  │         ▼                                                   │
  │  ┌───────────────────────┐   ┌───────────────────────┐      │
  │  │ enc_inputs.npy        │   │ query_only.npy        │      │
  │  │ dec_inputs.npy        │   │ tool_individual.npy   │      │
  │  │ dec_targets.npy       │   │ tool_ex_idx.npy       │      │
  │  │ loss_mask.npy         │   │ tool_is_pos.npy       │      │
  │  │ kept_idx.npy          │   │  (contrastive arrays) │      │
  │  │ tool_count.npy        │   └───────────────────────┘      │
  │  └──────┬────────────────┘                                  │
  │         ▼                                                   │
  │  {split}_metadata.json → .data_cache/                       │
  │  uploads tokenizer + arrays to GCS                          │
  └─────────────────────────────────────────────────────────────┘
                         │
                         ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  needle train                                               │
  │                                                             │
  │  load_prepared_data(mmap=True)                              │
  │       │                                                     │
  │       ▼                                                     │
  │  ┌──────────────────────┐   ┌──────────────────────┐        │
  │  │ PrefetchIterator     │   │ Contrastive batches  │        │
  │  │ text batches (4)     │   │ query-tool pairs     │        │
  │  │ mmap → per-batch idx │   │ CLIP in-batch neg    │        │
  │  └──────────┬───────────┘   └──────────┬───────────┘        │
  │             ▼                          ▼                    │
  │  text + contrastive tool-call training (jax.pmap)           │
  └─────────────────────────────────────────────────────────────┘
                         │
                         ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  needle calibrate                                           │
  │                                                             │
  │  Freeze encoder + decoder                                   │
  │  Compute per-example decoder perplexity                     │
  │  Map PPL → [0,1] confidence via sigmoid calibration         │
  │  Train confidence head (2 Dense layers, MSE loss)           │
  │  Save calibrated checkpoint with mu, k params               │
  └─────────────────────────────────────────────────────────────┘

  Hybrid Inference (confidence-based cloud routing)
  ─────────────────────────────────────────────────

  ┌─────────────────────────────────────────────────────────────┐
  │                                                             │
  │  query + tools                                              │
  │       │                                                     │
  │       ▼                                                     │
  │  ┌──────────┐                                               │
  │  │ Encoder  │                                               │
  │  └────┬─────┘                                               │
  │       │                                                     │
  │       ├──────────────────┐                                  │
  │       ▼                  ▼                                  │
  │  ┌──────────┐    ┌──────────────┐                           │
  │  │Confidence│    │  (encoder    │                           │
  │  │  Head    │    │   output)    │                           │
  │  └────┬─────┘    └──────┬───────┘                           │
  │       │                 │                                   │
  │       ▼                 │                                   │
  │  confidence > threshold?│                                   │
  │       │                 │                                   │
  │    ┌──┴──┐              │                                   │
  │    │ yes │──────────────┤                                   │
  │    └─────┘              ▼                                   │
  │                  ┌──────────┐                                │
  │                  │ Decoder  │──▶ tool calls (local)         │
  │                  └──────────┘                                │
  │    ┌─────┐                                                  │
  │    │ no  │──▶ route to cloud (Gemini) ──▶ tool calls        │
  │    └─────┘    (skip decoder entirely)                       │
  │                                                             │
  └─────────────────────────────────────────────────────────────┘
```

## Usage

```
git clone https://github.com/cactus-compute/needle.git

source ./setup

needle [command]

  ┌───────────────────────────────────────────────────────────────────┐
  │                                                                   │
  │   train                                                           │
  │     --epochs INT             Training epochs (default: 1)         │
  │     --batch-size INT         Batch size (default: 32)             │
  │     --lr FLOAT               AdamW learning rate (default: 3e-4)  │
  │     --muon-lr FLOAT          Muon learning rate (default: 0.02)   │
  │     --d-model INT            Model dim (default: 512)             │
  │     --num-heads INT          Attention heads (default: 8)         │
  │     --num-kv-heads INT       KV heads for GQA (default: 4)        │
  │     --num-layers INT         Encoder layers (default: 12)         │
  │     --num-dec-layers INT     Decoder layers (default: 12)         │
  │     --max-enc-len INT        Max encoder seq len (default: 1024)  │
  │     --max-dec-len INT        Max decoder seq len (default: 512)   │
  │     --max-samples INT        Training samples (default: all)      │
  │     --no-feedforward         No FFN layers (default: on)           │
  │     --feedforward            Enable FFN layers (off by default)   │
  │     --activation STR         swiglu|drelu|geglu (if FFN enabled)  │
  │     --mat-factors INT [...]  FFN shrink factors (if FFN enabled)  │
  │     --sparsity-ratio FLOAT   Block prune ratio (default: 0.0)     │
  │     --group-size INT         Quant/prune group size (default: 32) │
  │     --prune-interval INT     Steps between mask updates (def: 100)│
  │     --prune-start-frac FL    Start pruning at frac (def: 0.33)    │
  │     --prune-end-frac FL      Lock mask at this frac (def: 0.67)   │
  │     --warmup-ratio FLOAT     LR warmup ratio (default: 0.05)      │
  │     --decay-ratio FLOAT      LR cosine decay ratio (default: 0.05)│
  │     --dropout FLOAT          Dropout rate (default: 0.0)           │
  │     --eval-every INT         Val eval interval (default: 1000)    │
  │     --max-eval-samples INT   Val samples limit (default: 5000)    │
  │     --contrastive-weight FL  CLIP loss weight (default: 0.1)      │
  │     --contrastive-dim INT    Projection dim (default: 128)        │
  │     --w-name FLOAT           Loss weight: tool names (def: 2.0)   │
  │     --w-value FLOAT          Loss weight: arg values (def: 4.0)   │
  │     --w-key FLOAT            Loss weight: arg keys (def: 1.5)     │
  │     --wandb                  Enable W&B logging                   │
  │     --checkpoint PATH        Resume from checkpoint               │
  │     --checkpoint-dir DIR     Checkpoint directory                 │
  │     --dtype STR              float32|bfloat16 (default: bfloat16) │
  │     --seed INT               Random seed (default: 42)            │
  │                                                                   │
  │   tokenize                                                        │
  │     --max-samples INT        Limit samples per split (dev/test)   │
  │     --max-enc-len INT        Max encoder seq len (default: 1024)  │
  │     --max-dec-len INT        Max decoder seq len (default: 512)   │
  │     --shuffle-tools          Shuffle tool order (default: on)     │
  │     --no-shuffle-tools       Disable tool shuffling               │
  │     --max-tool-len INT       Max tool desc tokens (default: 256)  │
  │                                                                   │
  │   run                                                             │
  │     --checkpoint PATH        Path to model checkpoint (required)  │
  │     --query STR              Query text for tool-call generation  │
  │     --tools STR              Tools JSON for tool-call generation  │
  │     --audio PATH [...]       Audio files for voice-to-tool-call   │
  │     --max-len INT            Max tokens to generate (default: 512)│
  │     --seed INT               Random seed (default: 0)             │
  │     --no-constrained         Disable constrained decoding         │
  │                                                                   │
  │   eval                                                            │
  │     --checkpoint PATH        Path to model checkpoint (required)  │
  │     --batch-size INT         Batch size (default: 32)             │
  │     --max-eval-samples INT   Evaluation samples (default: 5000)   │
  │     --max-enc-len INT        Max encoder length (default: 1024)   │
  │     --max-dec-len INT        Max decoder length (default: 512)    │
  │     --max-gen-len INT        Max generation length (default: 512) │
  │     --throughput-runs INT    Throughput runs (default: 10)        │
  │     --tool-call-samples INT  Tool-call eval samples (default: 200)│
  │     --no-constrained         Disable constrained decoding         │
  │                                                                   │
  │   calibrate                                                       │
  │     --checkpoint PATH        Path to model checkpoint (required)  │
  │     --output PATH            Output path (default: overwrite)     │
  │     --batch-size INT         Batch size (default: 32)             │
  │     --num-samples INT        Limit samples for PPL (default: all) │
  │     --epochs INT             Training epochs (default: 10)        │
  │     --lr FLOAT               Learning rate (default: 1e-3)        │
  │     --k FLOAT                Sigmoid steepness (default: 3.0)     │
  │                                                                   │
  │   generate-data                                                   │
  │     --num-samples INT        Samples to generate (default: 5000)  │
  │     --batch-size INT         Examples per Gemini call (default:10)│
  │     --workers INT            Parallel Gemini calls (default: 8)   │
  │     --model STR              Gemini model override                │
  │     --dry-run                Generate only, skip upload           │
  │     --output-jsonl PATH      Also save raw generations to JSONL   │
  │     --upload-every INT       Merge+upload interval (def: 30000)   │
  │                                                                   │
  │   merge-xlam                                                      │
  │     --dry-run                Skip upload                          │
  │     --max-samples INT        Limit xlam samples                   │
  │                                                                   │
  │   rebalance-tools                                                 │
  │     --dry-run                Preview without modifying            │
  │                                                                   │
  │   split-dataset              Create train/val splits + upload     │
  │                                                                   │
  │   evaluate                                                        │
  │     --checkpoint PATH        Path to model checkpoint (required)  │
  │     --benchmarks [...]       wikitext2 lambada hellaswag arc_easy │
  │     --max-samples INT        Samples per benchmark (default: 500) │
  │                                                                   │
  │   tpu                                                             │
  │     create NAME              Create TPU (auto-finds zone)         │
  │       --type STR             Accelerator (default: v6e-8)         │
  │       --version STR          TPU OS (auto-detected from --type)   │
  │       --preemptible          Create spot/preemptible instance     │
  │     connect NAME             SSH config + connect (auto-zone)     │
  │     setup NAME               Full setup: sync code + venv + deps  │
  │     sync NAME                Fast sync: code (no venv rebuild)    │
  │     train NAME [-- ARGS]     Launch training on all workers       │
  │     claude NAME              Install Claude Code on instance      │
  │     stop NAME                Stop instance (auto-zone)            │
  │     start NAME               Start stopped instance (auto-zone)   │
  │     delete NAME              Delete instance (auto-zone)          │
  │     list                     List all TPU instances               │
  │       --zone ZONE            Override auto-detected zone          │
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
  Text                2M Tool-call pairs
                      (query, tools, answers)
                      synthesized from GCS
                      gs://cactus-dataset/
  ──────────────────────────────────────────

  ┌────────────────────┬──────────┬──────────┬──────────┐
  │                    │  v6e-8   │  v6e-16  │  v6e-32  │
  ├────────────────────┼──────────┼──────────┼──────────┤
  │ Chips              │ 8        │ 16       │ 32       │
  │ Total HBM          │ 256 GB   │ 512 GB   │ 1024 GB  │
  │ Scaling eff.       │ 0.9×     │ 0.8×     │ 0.7×     │
  │ Eff. TFLOPS        │ 994      │ 1,766    │ 3,091    │
  │ Est. time          │ ~88h     │ ~49h     │ ~29h     │
  │ On-demand $/hr     │ $21.60   │ $43.20   │ $86.40   │
  │ Est. total cost    │ ~$1,900  │ ~$2,120  │ ~$2,510  │
  └────────────────────┴──────────┴──────────┴──────────┘
```

## Setup For TPU/GCP

- Setup gcloud 1: download the `macOS ARM` from [here](https://docs.cloud.google.com/sdk/docs/install-sdk) and unzip.
- Setup gcloud 2: open terminal, cd to your downloads and run `./google-cloud-sdk/install.sh`
- Setup gcloud 3: restart terminal and run `gcloud init`, sign in with cactus email, should prompt for project
- Setup gcloud 4: else, set the project with `gcloud config set project needle-488623`
- Setup gcloud 5: run `gcloud help` and read carefully

## Example Workflow

### Single-host (v6e-8) — SSH into instance

```
1. Create an instance (auto: finds zone → installs Claude Code → connects via SSH)
   needle tpu create my-experiment
   (exit with 'exit' or Ctrl+D)

2. Reconnect anytime
   ssh my-experiment
   or VS Code: click the '><' in the bottom left → select my-experiment

--- run from the instance ---

3. Create a GitHub Personal Access Token (PAT)
   GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
   Generate a token with 'repo' scope

4. Clone the repo on your instance using your PAT
   git clone https://github.com/cactus-compute/needle.git
   cd needle

5. Install needle (follow instruction to setup wandb)
   source ./setup

6. Login to Hugging Face (required for private datasets)
   huggingface-cli login
   (paste your HF token — get one at https://huggingface.co/settings/tokens)

7. Full pipeline
   needle tokenize                            # tokenize + pack + upload
   needle train --wandb                       # train on TPU
   needle eval --checkpoint <path>            # evaluate

8. Use tmux for long training runs (survives SSH disconnects)
   tmux new -s train          # start a named session
   needle train --wandb       # run training inside it
   Ctrl+B, then D             # detach (keeps running)
   tmux attach -t train       # reattach later

--- back on your Mac ---

9. Stop when done (saves disk, stops billing)
   needle tpu stop my-experiment

10. (Optional) Delete instance when no longer needed
    needle tpu delete my-experiment
```

### Multi-host (v6e-16+) — run from your Mac

For multi-host TPUs (v6e-16 = 4 workers, v6e-32 = 8 workers), you drive
everything from your Mac. The CLI syncs code and launches training across
all workers automatically.

```
1. Set auth tokens (workers need these to download data + log metrics)
   export HF_TOKEN=hf_...
   export WANDB_API_KEY=...

2. Add SSH key to agent (required for gcloud scp/ssh)
   ssh-add ~/.ssh/google_compute_engine

3. Create a multi-host TPU (auto: finds zone → syncs code → installs deps)
   needle tpu create my-experiment --type v6e-16

4. Launch training on all workers from your Mac
   needle tpu train my-experiment -- --wandb --epochs 1

5. After code changes, sync without rebuilding venv (fast, ~seconds)
   needle tpu sync my-experiment

6. Full re-setup if deps changed (slow, rebuilds venv)
   needle tpu setup my-experiment

7. Stop/delete when done
   needle tpu stop my-experiment
   needle tpu delete my-experiment
```

## Current TPU Instances

```
  Name       Zone               Type    Software         Workers
  ──────────────────────────────────────────────────────────────────
  large      asia-northeast1-b  v6e-16  v2-alpha-tpuv6e  4
```
