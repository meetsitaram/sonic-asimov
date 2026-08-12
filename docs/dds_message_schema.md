# SONIC Asimov — policy I/O contract & DDS message schema

Audience: the team wiring SONIC into the Asimov robot's DDS bus. This
document specifies (1) the exact ONNX input/output tensor layouts, (2) all
orderings/conventions/units, and (3) a proposed set of DDS topics with IDL.
The reference implementation of everything in section 1–4 is
`scripts/eval_asimov_mujoco_onnx.py` — when in doubt, that file is the
ground truth (it is bit-exact against the IsaacLab training environment).

## 0. System overview

```
                     50 Hz                      50 Hz
 robot sensors ──► rt/sonic/robot_state ──► SONIC runner ──► rt/sonic/joint_cmd ──► motor driver
 (IMU + encoders)                        (obs build + ONNX)  (23 joint targets)     (PD ≥ 500 Hz)
                                              ▲
                              reference motion clip (PKL, 30 fps)
                              + internal playback clock
```

* The policy is a **motion-tracking** controller: it needs a reference clip
  streamed alongside proprioception. It does **not** need base position,
  linear velocity, or any odometry/mocap.
* One inference per **20 ms** tick (50 Hz). CPU onnxruntime inference is
  ~1–2 ms on a laptop core.
* The runner holds state between ticks: 10-frame observation history and
  the last action. Both reset when an episode (re)starts.

## 1. ONNX model contract

File: `models/locoft2_final_9800.onnx`

| | name | dtype/shape | meaning |
|---|---|---|---|
| input | `actor_obs` | float32 `(B, 1270)` | `[tokenizer_obs(520) | proprioception(750)]` |
| output | `action` | float32 `(B, 23)` | action mean, **IsaacLab DOF order** (§4) |

Internally: g1 encoder MLP `[520,2048,1024,512,512,64]` → FSQ quantizer →
decoder MLP `[64+750, 2048,2048,1024,1024,512,512,23]`, SiLU activations.
Deterministic (no sampling).

## 2. Input part A — `tokenizer_obs` (520 floats)

Reference-motion window: **10 future frames** at `t + k·0.1 s`,
`k = 0..9` (frame 0 = *current* reference frame). With 30 fps clips the
frame index is `base_frame + k*3` where `base_frame = round(t_clip * 30)`,
clamped to the last frame.

Build three blocks:

1. `jpos_ref[k][23]` — reference joint positions per future frame,
   **IsaacLab order**, **served convention** (§5). rad.
2. `jvel_ref[k][23]` — reference joint velocities: forward difference of
   served jpos × fps (last frame repeats the previous diff). rad/s.
3. `ori6[k][6]` — orientation error per future frame:
   `R_rel = R_base_now⁻¹ · R_ref[k]` (base orientation *now* vs reference
   root orientation at frame k), take the first two **columns** of the 3×3
   matrix, flattened **row-major**:
   `[R00, R01, R10, R11, R20, R21]`.

Then interleave — note the command block is NOT frame-major:

```
command_flat[460] = jpos_ref[0] ‖ jpos_ref[1] ‖ … ‖ jpos_ref[9]      (230)
                  ‖ jvel_ref[0] ‖ … ‖ jvel_ref[9]                    (230)

for i in 0..9:
    tokenizer_obs[52*i      : 52*i + 46] = command_flat[46*i : 46*(i+1)]
    tokenizer_obs[52*i + 46 : 52*i + 52] = ori6[i]
```

(i.e. the 460-float command vector is chunked into ten 46-float slices that
do not align with frame boundaries — this mirrors the training env's
`reshape(10, 46)` exactly. Do not "fix" it.)

## 3. Input part B — `proprioception` (750 floats)

Five terms, **term-major**, each with a 10-step history at 50 Hz
(**oldest first**; each step's vector contiguous):

| offset | term | per-step | frame/convention |
|---|---|---|---|
| 0 | `base_ang_vel` | 3 | body-frame angular velocity, rad/s (IMU gyro) |
| 30 | `joint_pos_rel` | 23 | `q − q_default` (§6), **raw encoder/qpos convention**, IL order, rad |
| 260 | `joint_vel` | 23 | joint velocity, IL order, rad/s |
| 490 | `last_action` | 23 | previous policy action **after clipping** (§7), IL order |
| 720 | `gravity_dir` | 3 | unit gravity in body frame = `R_base⁻¹ · [0,0,−1]` |
| 750 | *(end)* | | |

History semantics: on episode start the first sample **broadcast-fills all
10 slots**; afterwards it is a sliding window (append newest, drop oldest).
`last_action` starts as zeros.

## 4. Joint orderings (23 DOF)

**IsaacLab (IL) order — policy I/O (obs joints + action output):**

```
 0 left_hip_pitch    1 right_hip_pitch   2 waist_yaw
 3 left_hip_roll     4 right_hip_roll
 5 left_shoulder_pitch  6 right_shoulder_pitch
 7 left_hip_yaw      8 right_hip_yaw
 9 left_shoulder_roll  10 right_shoulder_roll
11 left_knee        12 right_knee
13 left_shoulder_yaw  14 right_shoulder_yaw
15 left_ankle_pitch  16 right_ankle_pitch
17 left_elbow       18 right_elbow
19 left_ankle_roll  20 right_ankle_roll
21 left_wrist_yaw   22 right_wrist_yaw
```

**MuJoCo/MJCF (motor) order — matches the MJCF joint tree:**

```
 0-5  left leg   (hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll)
 6-11 right leg  (same 6)
 12   waist_yaw
 13-17 RIGHT arm (shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_yaw)
 18-22 LEFT arm  (same 5)      ← note: right arm comes FIRST
```

Permutation (generated from the name lists — see
`IL_TO_MJ_DOF`/`MJ_TO_IL_DOF` in the player):

```
MJ_TO_IL = [0,3,7,11,15,19, 1,4,8,12,16,20, 2, 6,10,14,18,22, 5,9,13,17,21]
arr_mj = arr_il[MJ_TO_IL]        arr_il = arr_mj[IL_TO_MJ]
```

Quaternions: DDS messages below use **wxyz**. (The motion PKL stores root
rotation as xyzw — the runner converts.)

## 5. The two joint-angle conventions (critical)

There are **two different joint-angle conventions** in this contract:

* **Raw convention** — what joint encoders / MuJoCo qpos report. Used for
  ALL proprioception (§3) and for the action→target mapping (§7).
* **Served convention** — used ONLY for the reference streams in the
  tokenizer (§2 `jpos_ref`/`jvel_ref`). The training motion library
  re-extracts joint angles from rotation matrices with per-axis atan2 and
  the |axis| convention; net effect vs raw: every **negative-axis joint is
  sign-flipped** (the sign-mirrored right side, the yaw axes, ankle/
  shoulder rolls) and the two canted wrist joints get a nonlinear remap.

The bundled PKL's `dof` stream is raw; the player derives the served stream
from `pose_aa` (`_served_dof_from_pose_aa` in the player — ~20 lines, port
as-is). Feeding raw reference joints instead makes the policy track a ghost
target and collapse into a stiff default-pose gait (this was the sim2sim
bug of 2026-07-28; joint MAE 0.29 vs the healthy 0.17).

If you stream references from these PKLs on-robot, either port that
function or precompute `_dof_served`/`_dof_vel_served` offline per clip.

## 6. Default pose (rad, raw convention)

Nonzero entries only; all other joints 0:

```
left_hip_pitch +0.10   right_hip_pitch −0.10
left_shoulder_pitch −0.35   right_shoulder_pitch +0.35
left_shoulder_roll  −0.18   right_shoulder_roll  +0.18
left_elbow +0.87            right_elbow −0.87
```

Used both for `joint_pos_rel` (§3) and the action target (§7).

## 7. Output → motor command

Per 50 Hz tick, from the raw ONNX output `a_il[23]`:

```
a_il      = clip(a_il, −20, +20)          # this clipped value is also what
                                          # enters the last_action history
a_mj      = a_il[MJ_TO_IL]                # reorder to motor order
q_target  = q_default + a_mj * action_scale     # rad, raw convention
```

`action_scale[j] = 0.25 * effort_j / kp_j` (per joint). The motor driver
runs plain PD around `q_target` (zero velocity target, no feedforward):

```
τ = kp * (q_target − q) − kd * q̇          clamped to ±τ_max
```

Per-joint table (kp, kd, and the **saturation** effort the policy was
trained with — the sim clamps torque at these):

| joint (basename) | kp | kd | effort τ_sat [Nm] | action_scale [rad] |
|---|---|---|---|---|
| hip_pitch | 150 | 5 | 120 | 0.200 |
| hip_roll | 150 | 5 | 90 | 0.150 |
| hip_yaw | 150 | 5 | 60 | 0.100 |
| knee | 150 | 5 | 75 | 0.125 |
| ankle_pitch | 440 | 20 | 145.4 | 0.0826 |
| ankle_roll | 440 | 20 | 57.6 | 0.0327 |
| waist_yaw | 65 | 5 | 120 | 0.4615 |
| shoulder_pitch | 57 | 5 | 90 | 0.3947 |
| shoulder_roll | 86 | 5 | 75 | 0.2180 |
| shoulder_yaw | 96 | 5 | 60 | 0.1563 |
| elbow | 40 | 2 | 36 | 0.225 |
| wrist_yaw | 40 | 2 | 36 | 0.225 |

**Safety limits (bare minimum, deploy-mirrored).** Two output-level clamps,
applied *downstream* of the network (the `last_action` history in §3 must
keep the training-convention ±20 clip, NOT these):

```
|action|                 <= 12      per element   (nominal walk peaks ~9.7)
|q_target − q_default|   <= 2.2 rad per joint     (nominal walk peaks ~1.52)
```

The player applies both by default (`--max-action 12 --max-dev 2.2`, 0
disables) and reports engagement counts at exit. They are sized to never
engage in nominal tracking — **any engagement in normal operation is a
fault signal** (policy blow-up, bad reference, or convention bug), so wire
the counts into `PolicyStatus`-style telemetry.

Deploy note (X2 lesson): keep **peak/saturation** limits in any sim, but
size the *hardware* safety clamps from the continuous ratings — the driver
may clamp lower than τ_sat for thermal safety without retraining, at some
cost to tracking of fast transients.

The sim reference applies PD at 200 Hz (4× decimation of the 50 Hz action);
on hardware run the PD at whatever rate the driver natively supports
(≥500 Hz recommended), holding `q_target` between policy ticks.

## 8. Episode start (RSI) & safety terminations

On start, the robot should be posed near the reference's first frame
(joint targets = served frame-0 joints clamped to 0.9-soft limits; the sim
also sets root pose/velocity — on hardware, start clips from a
standing-idle first frame). Reset the obs history and `last_action`, and
start the clip clock at 0.

The eval harness terminates on `pelvis_z < 0.35 m` (Asimov stands ~0.63 m)
or base tilt > ~72°; wire equivalent watchdogs into the deploy stack.

## 9. Proposed DDS topics

Domain/QoS suggestion: RELIABLE, KEEP_LAST(1), sensor topics VOLATILE.
All arrays fixed-size; all joint arrays in **motor (MJCF) order** on the
wire — the runner owns the IL permutation, so drivers never see IL order.

```idl
module sonic {

  // robot -> runner, >= 50 Hz (runner samples latest at each tick)
  struct RobotState {
    unsigned long long stamp_ns;
    float base_quat_wxyz[4];   // IMU orientation, world frame, wxyz
    float base_ang_vel[3];     // gyro, BODY frame, rad/s
    float joint_pos[23];       // motor order, rad, raw encoder convention
    float joint_vel[23];       // motor order, rad/s
  };

  // runner -> motor driver, 50 Hz
  struct JointCommand {
    unsigned long long stamp_ns;
    unsigned long seq;         // policy tick counter
    float q_target[23];        // motor order, rad (raw convention)
    float kp[23];              // table of section 7 (constant today)
    float kd[23];
    float tau_max[23];         // driver-side torque clamp
  };

  // runner -> monitoring (optional)
  struct PolicyStatus {
    unsigned long long stamp_ns;
    unsigned long seq;
    string  clip_name;
    float   clip_time_s;       // reference playback clock
    float   action_inf_norm;   // pre-clip |a|_inf, saturation telltale
    boolean fallen;            // watchdog latched (section 8)
  };
};
```

Runner responsibilities per tick: sample newest `RobotState`; derive
`gravity_dir = R(base_quat)⁻¹·[0,0,−1]`; permute joints to IL order;
maintain the 10-step histories; advance the clip clock by 20 ms; build the
1270-float obs (§2–3); run ONNX; apply §7; publish `JointCommand`.

If you prefer to keep the runner off the realtime bus, an alternative
split is `JointCommand.q_target` only + static kp/kd/τ_max in driver
config — the schema above just makes the contract explicit on the wire.

## 10. Reference motion format (for streaming your own clips)

`motions/*.pkl` = joblib dict `{clip_name: clip}`, each clip:

| key | shape | meaning |
|---|---|---|
| `fps` | scalar | 30 |
| `dof` | (T, 23) | joint angles, motor order, **raw** convention, rad |
| `pose_aa` | (T, 24, 3) | axis-angle per body (root + 23 joints) — source for the served convention |
| `root_trans_offset` | (T, 3) | root position, world, m |
| `root_rot` | (T, 4) | root quaternion, **xyzw** |
