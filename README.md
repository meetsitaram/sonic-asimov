# sonic-asimov

Standalone bundle for running the **SONIC whole-body tracking policy on the
Asimov v1 humanoid (23 actuated DOF, Menlo Research) in MuJoCo**, tracking a
relaxed-walk reference motion. Everything needed is in this repo: policy
(ONNX), robot model (MJCF + meshes), reference motions, player script, and
an install script. CPU-only, no GPU and no torch required.

## Quickstart

```bash
./install.sh            # creates .venv with mujoco + onnxruntime (+ scipy/joblib/numpy)
./play_relaxed_walk.sh  # MuJoCo viewer: SONIC tracks the relaxed-walk clip
```

Useful variants:

```bash
./play_relaxed_walk.sh --clip impro                      # any of the 24 clips (substring match)
./play_relaxed_walk.sh --no-viewer --total-sim-seconds 20  # headless smoke test + metrics
./play_relaxed_walk.sh --show-forces                     # draw contact forces
```

Viewer keys: `SPACE` pause · `R` reset · `N` next clip · `V` toggle
tracking/free camera · arrow keys = 80 N push at the torso
(front/back/left/right).

Every episode prints a tracking summary (survival seconds, joint MAE in rad,
pelvis-height MAE). A healthy run tracks the walk at **joint MAE ≈ 0.17 rad**
and survives to `motion_end`; MAE ≈ 0.29 with a stiff default-pose gait means
the reference convention is broken (see the "served DOF" note in the player).

## Contents

| Path | What |
|---|---|
| `models/locoft2_final_9800.onnx` | Latest SONIC Asimov policy (fused g1-encoder → FSQ → decoder, single 1270-D input → 23-D action). |
| `models/locoft2_final_config.yaml` | Resolved training config for that checkpoint (reference). |
| `motions/asimov_relaxed_walk.pkl` | 24 relaxed-walk clips (`walk_forward_relax_*`), 30 fps, retargeted G1→Asimov. In-distribution for this policy. |
| `assets/mjcf/asimov.xml` (+ `asimov_assets/`) | Asimov v1 MuJoCo model. |
| `scripts/eval_asimov_mujoco_onnx.py` | The player: builds observations exactly as in training, runs the ONNX at 50 Hz, PD at 200 Hz. |
| `scripts/export_asimov_onnx.py` | Reference: how the ONNX was exported from the .pt checkpoint (needs torch; not needed for playback). |
| `tools/make_motion_pkl.py` | Provenance: how the motion PKL was extracted from the training corpus. |
| `docs/dds_message_schema.md` | **Integration contract**: exact policy input/output layout + proposed DDS topics/IDL. |

## Model lineage

`locoft2_final_9800` (2026-07-28): SONIC universal-token tracking policy,
trained in IsaacLab on the Asimov loco2k corpus (2,173 locomotion clips
retargeted from G1 bones-seed motions), warm-started after the
**saturation-effort fix** — sim effort limits are the actuators' *peak*
ratings, not continuous/thermal ones (with continuous limits the policy
physically could not bend knees). The player's PD/effort tables match that
training config verbatim; do not evaluate older (`bigrun`/`locoft1`)
checkpoints without `--legacy-continuous-efforts`.

ONNX↔checkpoint parity was gated at export (max action diff ~1e-6) and can
be re-checked anytime with `--compare-pt <ckpt.pt>` (needs torch).

## Integrating on the robot

Read `docs/dds_message_schema.md`. Short version: publish IMU gyro +
orientation and joint states at ≥50 Hz, run the ONNX once per 20 ms tick
with the observation vector specified there, and send the resulting 23
target joint positions to a motor-side PD loop with the kp/kd table from the
doc. The policy needs **no odometry** — only gyro, gravity direction, joint
pos/vel, its own last action, and the reference-clip playback clock.
