#!/usr/bin/env bash
# Launches the ablation runner in a persistent tmux session.
# Usage: bash ablations/launch_tmux.sh [SESSION_NAME]
#
# After launching:
#   tmux attach -t ablations          # watch live output
#   Ctrl-B D                          # detach (leave running)
#   python ablations/runner.py --list # check status from any shell

set -euo pipefail
cd "$(dirname "$0")/.."

SESSION="${1:-ablations}"

# Load credentials into the environment the runner will inherit
source hf_token.sh

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Session '$SESSION' already exists."
    echo "  Attach:  tmux attach -t $SESSION"
    echo "  Status:  python ablations/runner.py --list"
    exit 0
fi

tmux new-session -d -s "$SESSION" -x 220 -y 50

# Source credentials inside tmux, then launch runner
tmux send-keys -t "$SESSION" "source hf_token.sh && python ablations/runner.py" Enter

echo "Launched ablation runner in tmux session '$SESSION'."
echo ""
echo "  Watch:   tmux attach -t $SESSION"
echo "  Detach:  Ctrl-B then D"
echo "  Status:  python ablations/runner.py --list"
echo "  W&B:     https://wandb.ai (project: needle-pretrain)"
