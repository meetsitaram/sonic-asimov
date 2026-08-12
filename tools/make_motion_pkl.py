#!/usr/bin/env python3
"""Provenance: how motions/asimov_relaxed_walk.pkl was built.

Extracts the walk_forward_relax_* family from the full Asimov loco2k
training corpus (asimov_loco_2k.pkl, 2,173 clips — G1 bones-seed motions
retargeted to Asimov v1). Drops the `smpl_joints` stream (unused by the
MuJoCo player; keeps root_trans_offset / pose_aa / dof / root_rot / fps —
`pose_aa` is REQUIRED: the served-DOF reference convention is derived
from it, see the note in scripts/eval_asimov_mujoco_onnx.py).

Run from the source repo (GR00T-WholeBodyControl):
    python tools/make_motion_pkl.py \
        --corpus gear_sonic/data/motions/asimov_loco_2k.pkl \
        --out motions/asimov_relaxed_walk.pkl
"""
import argparse

import joblib

KEEP_FIELDS = ("root_trans_offset", "pose_aa", "dof", "root_rot", "fps")
PREFIX = "walk_forward_relax"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", required=True, help="Full corpus PKL (joblib).")
    ap.add_argument("--out", required=True, help="Output PKL path.")
    ap.add_argument("--prefix", default=PREFIX)
    args = ap.parse_args()

    corpus = joblib.load(args.corpus)
    slim = {
        name: {f: clip[f] for f in KEEP_FIELDS}
        for name, clip in corpus.items()
        if name.startswith(args.prefix)
    }
    if not slim:
        raise SystemExit(f"No clips matching prefix '{args.prefix}'")
    joblib.dump(slim, args.out)
    print(f"Wrote {len(slim)} clips -> {args.out}")
    for name in sorted(slim):
        m = slim[name]
        print(f"  {name}: {m['dof'].shape[0]} frames @ {m['fps']:g} fps")


if __name__ == "__main__":
    main()
