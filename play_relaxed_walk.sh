#!/usr/bin/env bash
# Play the relaxed-walk reference motion in MuJoCo with the SONIC policy
# driving the robot (ONNX inference, CPU).
#
#   ./play_relaxed_walk.sh                 # default clip, viewer
#   ./play_relaxed_walk.sh --clip impro    # any clip whose name contains "impro"
#   ./play_relaxed_walk.sh --no-viewer --total-sim-seconds 20   # headless
#   ./play_relaxed_walk.sh --kinematic     # reference motion only (no physics/policy)
#
# Viewer keys: SPACE pause | R reset | N next clip | V camera |
#              arrow keys = 80 N push (front/back/left/right)
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ]; then
    echo "No .venv found — run ./install.sh first." >&2
    exit 1
fi

ARGS=("$@")
# Default to the reference relaxed-walk clip unless the caller picked one.
if [[ ! " $* " =~ " --clip " ]]; then
    ARGS+=(--clip walk_forward_relax_001__A002)
fi

exec .venv/bin/python scripts/eval_asimov_mujoco_onnx.py \
    --onnx models/locoft2_final_9800.onnx \
    --motion motions/asimov_relaxed_walk.pkl \
    "${ARGS[@]}"
