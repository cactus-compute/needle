#!/usr/bin/env bash
set -euo pipefail

# NAR experiment sweep — run sequentially on the VM
# Each experiment changes one variable from the baseline NAR config
# Baseline: --nar --num-queries 20 --nar-checkpoint AR_CKPT --epochs 1

AR_CKPT="checkpoints/needle_4_1024_8279.pkl"
COMMON="--nar --num-queries 20 --nar-checkpoint $AR_CKPT --epochs 1 --seed 42 --sparsity-ratio 0.0 --eval-every 500 --wandb"

echo "========================================"
echo "NAR Experiment 1: Higher LR (10x)"
echo "========================================"
needle train $COMMON --lr 3e-3 --checkpoint-dir checkpoints/nar_exp1_highlr
echo ""

echo "========================================"
echo "NAR Experiment 2: Orthogonal query init"
echo "========================================"
needle train $COMMON --query-init orthogonal --checkpoint-dir checkpoints/nar_exp2_ortho
echo ""

echo "========================================"
echo "NAR Experiment 3: AdamW only (no Muon)"
echo "========================================"
needle train $COMMON --adamw-only --checkpoint-dir checkpoints/nar_exp3_adamw
echo ""

echo "========================================"
echo "NAR Experiment 4: Sliding window pairs"
echo "========================================"
needle train $COMMON --nar-sliding --checkpoint-dir checkpoints/nar_exp4_sliding
echo ""

echo "All experiments complete."
