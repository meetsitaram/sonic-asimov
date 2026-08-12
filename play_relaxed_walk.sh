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
# Default: the relaxed-walk reference clip, started at frame 70 — the
# clip's first ~2.3 s is the from-rest acceleration (hands held folded
# forward, torso leaned in); by f70 the arms have dropped to a natural
# hang and the gait is settled. NOTE: the walk_forward_relax_* clips
# instead open with a ~2.5 s T-pose calibration ramp — pass
# --init-frame 75 with those.
if [[ ! " $* " =~ " --clip " ]]; then
    ARGS+=(--clip Relaxed_walk_forward_001__A057)
    if [[ ! " $* " =~ " --init-frame " ]]; then
        ARGS+=(--init-frame 70)
    fi
fi

exec .venv/bin/python scripts/eval_asimov_mujoco_onnx.py \
    --onnx models/locoft2_final_9800.onnx \
    --motion motions/asimov_relaxed_walk.pkl \
    "${ARGS[@]}"
