# sonic-asimov

Standalone bundle for running the **SONIC whole-body tracking policy on the
Asimov v1 humanoid (23 actuated DOF, Menlo Research) in MuJoCo**, tracking a
relaxed-walk reference motion. Everything needed is in this repo: policy
(ONNX), robot model (MJCF + meshes), reference motions, player script, and
an install script. CPU-only, no GPU and no torch required.

## Kinematic reference vs SONIC (side by side)

`Relaxed_walk_forward_001__A057` (14.8 s, starts from rest) — left: the
reference motion played kinematically (poses written straight to the sim,
no physics); right: the SONIC policy tracking that same reference under
full physics (joint MAE 0.156 rad, no falls, safety limits never engaged):

![Kinematic vs SONIC preview](media/relaxed_walk_preview.gif)

*(12 s preview — full-length videos:
[side-by-side](media/relaxed_walk_side_by_side.mp4) ·
[kinematic](media/relaxed_walk_kinematic.mp4) ·
[SONIC](media/relaxed_walk_sonic.mp4))*

> The full-length `.mp4`s are tracked with **git LFS** — install `git-lfs`
> before cloning to get them. Without it you get harmless pointer stubs; the
> GIF preview above and everything else (model, motions, scripts) is plain
> git and works regardless.

Reproduce with the player's `--record` flag (needs ffmpeg on PATH):

```bash
.venv/bin/python scripts/eval_asimov_mujoco_onnx.py --kinematic --no-viewer \
    --motion motions/asimov_motions.pkl --clip Relaxed_walk_forward_001__A057 \
    --record media/relaxed_walk_kinematic.mp4
.venv/bin/python scripts/eval_asimov_mujoco_onnx.py --onnx models/locoft2_final_9800.onnx \
    --motion motions/asimov_motions.pkl --clip Relaxed_walk_forward_001__A057 \
    --no-viewer --max-episode 20 --record media/relaxed_walk_sonic.mp4
```

## Quickstart

```bash
./install.sh            # creates .venv with mujoco + onnxruntime (+ scipy/joblib/numpy)
./play_relaxed_walk.sh  # MuJoCo viewer: SONIC tracks the relaxed-walk clip
```

Useful variants:

```bash
./play_relaxed_walk.sh --clip neutral_idle               # idle-stand loop (stand in place)
./play_relaxed_walk.sh --clip impro                      # any of the 27 clips (substring match)
./play_relaxed_walk.sh --no-viewer --total-sim-seconds 20  # headless smoke test + metrics
./play_relaxed_walk.sh --show-forces                     # draw contact forces
./play_relaxed_walk.sh --kinematic                       # reference motion only, no physics/policy
```

Viewer keys: `SPACE` pause · `R` reset · `N` next clip · `V` toggle
tracking/free camera · arrow keys = 80 N push at the torso
(front/back/left/right).

Deploy-style output safety limits are ON by default — `--max-action 12`
(action cap; training clip is ±20) and `--max-dev 2.2` (per-joint target
deviation from default pose, rad). Both are sized to never engage in
nominal walking (measured peaks: 9.7 / 1.52 rad); engagement counts print
at exit, and a nonzero count is a fault signal. Pass `0` to disable.

Every episode prints a tracking summary (survival seconds, joint MAE in rad,
pelvis-height MAE). A healthy run tracks the walk at **joint MAE ≈ 0.17 rad**
and survives to `motion_end`; MAE ≈ 0.29 with a stiff default-pose gait means
the reference convention is broken (see the "served DOF" note in the player).

## Contents

| Path | What |
|---|---|
| `models/locoft2_final_9800.onnx` | Latest SONIC Asimov policy (fused g1-encoder → FSQ → decoder, single 1270-D input → 23-D action). |
| `models/locoft2_final_config.yaml` | Resolved training config for that checkpoint (reference). |
| `motions/asimov_motions.pkl` | 27 clips @ 30 fps, retargeted G1→Asimov: `neutral_idle_loop_001__A074` (idle stand — the natural episode-start/handoff reference), `Relaxed_walk_forward_001__A057` (+ mirror; the demo clip — the play script starts it at frame 70, after the from-rest acceleration where the hands hang forward) and the `walk_forward_relax_*` family (those open with a ~2.5 s T-pose mocap-calibration ramp — play them with `--init-frame 75`). |
| `assets/mjcf/asimov.xml` (+ `asimov_assets/`) | Asimov v1 MuJoCo model. |
| `scripts/eval_asimov_mujoco_onnx.py` | The player: builds observations exactly as in training, runs the ONNX at 50 Hz, PD at 200 Hz. |
| `docs/dds_message_schema.md` | **Integration contract**: exact policy input/output layout + proposed DDS topics/IDL. |

## Model

SONIC universal-token tracking policy, trained on ~2k locomotion clips
retargeted to Asimov from the bones-seed G1 motion dataset. The player's
PD gains and effort limits match the training config verbatim, and
ONNX↔checkpoint parity was gated at export time (max action diff ~1e-6).

## Integrating on the robot

Read `docs/dds_message_schema.md`. Short version: publish IMU gyro +
orientation and joint states at ≥50 Hz, run the ONNX once per 20 ms tick
with the observation vector specified there, and send the resulting 23
target joint positions to a motor-side PD loop with the kp/kd table from the
doc. The policy needs **no odometry** — only gyro, gravity direction, joint
pos/vel, its own last action, and the reference-clip playback clock.
