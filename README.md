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

                    ┌──────────────┐
                    │  Tool Call   │                ┌──────────────────────┐
                    └──────┬───────┘                │ Mat loss at factors  │
                          ┌┴──────────┐             │ + INT4 QAT (g=32)    │
                          │  Softmax  │             │ + z-loss + slot div  │
                          └─────┬─────┘             │ 1:1 text/speech     │
                    ┌─────┬─────┴─────┬─────┐       │ alternation          │
                    │  @E[:d_ff/f] for each │       └──────────┬───────────┘
                    │  f in mat_factors     │                  │
                    └─────┬─────┬─────┬─────┘       ┌──────────┴───────────┐
                          ┌─────┴─────┐             │ Grad clip (norm 1.0) │
                          │ Linear (T)│  ← tied     │ Muon  (2D kernels)   │
                          └─────┬─────┘             │ AdamW (all else)     │
                          ┌─────┴─────┐             │ WSD LR schedule      │
                          │ ZCRMSNorm │             └──────────┬───────────┘
                          └─────┬─────┘                        │
                       ┌────────┴────────┐          ┌──────────┴───────────┐
                       │ Decoder x N_dec │          │   Sparsification     │
                       │┌───────────────┐│          │  cubic schedule      │
                       ││ Masked Self   ││          │  mask every N steps  │
                       ││ Attn + RoPE   ││          └──────────┬───────────┘
                       │├───────────────┤│                     │
  ┌──────────────┐  S  ││   Cross       ││          ┌──────────┴───────────┐
  │ MemoryMixer  │────────▶ Attention   ││          │ EMA params (β=0.999) │
  │ Encoder      │     │├───────────────┤│          └──────────┬───────────┘
  │  x N_enc     │     ││  Gated FFN    ││                     │
  │              │     │└───────────────┘│          ┌──────────┴───────────┐
  │ ┌──────────┐ │     └────────┬────────┘          │ Checkpoint           │
  │ │Pack:     │ │        ┌─────┴─────┐             │  full params, can    │
  │ │ S←X Attn │ │        │ Embedding │  ← shared   │  export mat slices    │
  │ │ RoPE keys│ │        └─────┬─────┘             └──────────────────────┘
  │ ├──────────┤ │      ┌───────┴───────┐
  │ │Mix:      │ │      │[EOS]<tool_call>│
  │ │ MLP-Mixer│ │      │ + tools       │
  │ │ on S     │ │      │ + answer      │
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

    d=512 (default) / 1536 (--full) · GQA · QK-norm
    SentencePiece BPE (8192) · gated dReLU · ZCRMSNorm · RoPE
    strided DW conv · mat factors · INT4 QAT · Muon + AdamW
    text + speech encoder · <tool_call> task routing

  Data Preprocessing
  ──────────────────

  Text (Tool-Call Dataset)                 Speech (Voice-Tool-Call)
  ────────────────────────                 ────────────────────────

     Unified dataset                         Unified dataset
     (GCS / local)                           (GCS / local)
         │                                       │
         ▼                                       ▼
  ┌──────────────┐                      ┌────────────────┐
  │ SentencePiece│                      │ HF Audio col   │
  │ BPE tokenize │                      │ decode (16kHz)  │
  │ (8192 vocab) │                      └───────┬────────┘
  └──────┬───────┘                              │
         │                                      ▼
         ▼                              ┌────────────────┐
  ┌──────────────┐                      │ Augmentation   │
  │ Encoder:     │                      │ (noise, pitch, │
  │ query tokens │                      │  stretch, gain)│
  └──────┬───────┘                      └───────┬────────┘
         │                                      │
         ▼                                      ▼
  ┌──────────────────┐                  ┌────────────────┐
  │ Decoder:         │                  │ Log-mel spec   │
  │ [EOS]<tool_call> │                  │ 80 bins, 25ms  │
  │ + tools tokens   │                  │ window, 10ms   │
  │ + answer tokens  │                  │ hop (~100 f/s) │
  │ → dec_in         │                  └───────┬────────┘
  ├──────────────────┤                          │
  │ <tool_call>      │                          ▼
  │ + tools tokens   │                  ┌────────────────┐
  │ + answer tokens  │                  │ Pad/trunc to   │
  │ + [EOS]          │                  │ max_mel_len    │
  │ → dec_tgt        │                  └───────┬────────┘
  ├──────────────────┤                          │
  │ Loss mask: 1.0   │                          ▼
  │ on answer + EOS  │                  ┌────────────────┐
  │ only (not tools) │                  │ Same decoder   │
  └──────────────────┘                  │ format as text │
                                        └────────────────┘
         Both cached as
         .npy in .data_cache/           Mel computed on-the-fly
                                        (augment before mel)
  ┌──────────────────────┐
  │ Interleaved batches: │
  │ 1:1 text / speech    │       Eval: voice-to-tool-call on
  │ alternation          │       3 val samples per epoch
  └──────────────────────┘
```

## Usage

```
git clone https://github.com/cactus-compute/needle.git

source ./setup

needle [command]

  ┌───────────────────────────────────────────────────────────────────┐
  │                                                                   │
  │   train                                                           │
  │     --full                   Use full 1B config (~1.17B params)   │
  │     --epochs INT             Training epochs (default: 1)         │
  │     --batch-size INT         Batch size (default: 32)             │
  │     --lr FLOAT               AdamW learning rate (default: 3e-4)  │
  │     --muon-lr FLOAT          Muon learning rate (default: 0.02)   │
  │     --d-model INT            Model dim (default: 512)             │
  │     --num-heads INT          Attention heads (default: 8)         │
  │     --num-kv-heads INT       KV heads for GQA (default: num-heads)│
  │     --num-layers INT         Encoder layers (default: 4)          │
  │     --num-dec-layers INT     Decoder layers (default: 4)          │
  │     --max-enc-len INT        Max encoder seq len (default: 256)   │
  │     --max-dec-len INT        Max decoder seq len (default: 1024)  │
  │     --max-samples INT        Training samples (default: all)      │
  │     --mat-factors INT [...]   FFN shrink factors (default: 2 4 8) │
  │     --sparsity-ratio FLOAT   Block prune ratio (default: 0.0)     │
  │     --group-size INT         Quant/prune group size (default: 32) │
  │     --prune-interval INT     Steps between mask updates (def: 100)│
  │     --prune-start-frac FL    Start pruning at frac (def: 0.33)    │
  │     --prune-end-frac FL      Lock mask at this frac (def: 0.67)   │
  │     --activation STR         drelu|swiglu|geglu (default: drelu)  │
  │     --warmup-ratio FLOAT     LR warmup ratio (default: 0.05)      │
  │     --eval-every INT         Val eval interval (default: 1000)    │
  │     --wandb                  Enable W&B logging                   │
  │     --checkpoint PATH        Resume from checkpoint               │
  │     --checkpoint-dir DIR     Checkpoint directory                 │
  │     --seed INT               Random seed (default: 42)            │
  │     --no-speech             Disable speech (text-only training)   │
  │     --max-mel-len INT       Max mel frames (default: 1024)        │
  │     --n-mels INT            Mel frequency bins (default: 80)      │
  │     --max-speech-samples INT  Max voice-tool-call samples         │
  │                                                                   │
  │   run                                                             │
  │     --checkpoint PATH       Path to model checkpoint (required)   │
  │     --query STR             Query text for tool-call generation   │
  │     --tools STR             Tools JSON for tool-call generation   │
  │     --audio PATH [...]      Audio files for voice-to-tool-call    │
  │     --max-len INT           Max tokens to generate (default: 512) │
  │     --seed INT              Random seed (default: 0)              │
  │                                                                   │
  │   test                                                            │
  │     --checkpoint PATH       Path to model checkpoint (required)   │
  │     --batch-size INT        Batch size (default: 32)              │
  │     --max-eval-samples INT  Evaluation samples (default: 1000)    │
  │     --max-enc-len INT       Max encoder length (default: 256)    │
  │     --max-dec-len INT       Max decoder length (default: 1024)    │
  │     --max-gen-len INT       Max generation length (default: 512)  │
  │     --throughput-runs INT   Throughput runs (default: 10)         │
  │     --tool-call-samples INT Tool-call eval samples (default: 200) │
  │     --voice-tc-samples INT  Voice-TC eval samples (default: 50)  │
  │                                                                   │
  │   evaluate                                                        │
  │     --checkpoint PATH       Path to model checkpoint (required)   │
  │     --benchmarks [...]      wikitext2 lambada hellaswag arc_easy  │
  │     --max-samples INT       Samples per benchmark (default: 500)  │
  │                                                                   │
  │   tpu                                                             │
  │     create NAME             Create TPU (auto-finds zone)          │
  │       --type STR            Accelerator (default: v6e-8)          │
  │       --version STR         TPU OS (auto-detected from --type)    │
  │     connect NAME            SSH config + connect (auto-zone)      │
  │     claude NAME             Install Claude Code on instance       │
  │     stop NAME               Stop instance (auto-zone)             │
  │     start NAME              Start stopped instance (auto-zone)    │
  │     delete NAME             Delete instance (auto-zone)           │
  │     list                    List all TPU instances                │
  │       --zone ZONE           Override auto-detected zone           │
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
  Text                Tool-call pairs
                      (query, tools, answers)
                      from GCS unified dataset
  Audio               TTS-generated speech
                      (GCP Cloud TTS voices)
                      16kHz WAV, HF Audio col
  ──────────────────────────────────────────

  100M params — 10M multimodal triples (5 epochs)
  ──────────────────────────────────────────
  Triples          10M (text / voice / image → JSON tool calls)
  Tokens/triple    ~300 avg (text ~180, voice ~390, image ~336)
  Total tokens     10M × 300 × 5 epochs ≈ 15B
  Total FLOPs      6 × 100M × 15B ≈ 9e18
  MFU (100M)       ~15% → ~138 effective TFLOPS/chip

  ┌────────────────────┬──────────┬──────────┬──────────┐
  │                    │  v6e-8   │  v6e-16  │  v6e-32  │
  ├────────────────────┼──────────┼──────────┼──────────┤
  │ Chips              │ 8        │ 16       │ 32       │
  │ Total HBM          │ 256 GB   │ 512 GB   │ 1024 GB  │
  │ Scaling eff.       │ 0.9×     │ 0.8×     │ 0.7×     │
  │ Eff. TFLOPS        │ 994      │ 1,766    │ 3,091    │
  │ Est. time          │ ~2.5h    │ ~1.4h    │ ~49min   │
  │ On-demand $/hr     │ $21.60   │ $43.20   │ $86.40   │
  │ Est. total cost    │ ~$54     │ ~$61     │ ~$71     │
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
