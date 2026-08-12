#!/usr/bin/env bash
# Create a local venv with everything needed to run the SONIC Asimov
# MuJoCo player (ONNX-only path — no torch required).
#
# Requires: python3.10+ with the venv module, internet access.
# Usage:    ./install.sh
set -euo pipefail
cd "$(dirname "$0")"

PY=${PYTHON:-python3}
echo "Using interpreter: $($PY --version)"

$PY -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install \
    "mujoco>=3.1" \
    "onnxruntime>=1.17" \
    "numpy<2" \
    scipy \
    joblib

echo
echo "Install OK. Smoke-checking imports..."
.venv/bin/python - <<'EOF'
import mujoco, onnxruntime, numpy, scipy, joblib
print("mujoco", mujoco.__version__, "| onnxruntime", onnxruntime.__version__,
      "| numpy", numpy.__version__)
model = mujoco.MjModel.from_xml_path("assets/mjcf/asimov.xml")
print("Asimov MJCF loads:", model.njnt, "joints,", model.nu, "actuators")
EOF

echo
echo "Done. Run ./play_relaxed_walk.sh to launch the viewer."
