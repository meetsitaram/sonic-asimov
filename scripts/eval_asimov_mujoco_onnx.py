#!/usr/bin/env python3
"""ONNX-driven MuJoCo evaluation for Asimov v1 (23 actuated DOF, Menlo
Research). Point it at the fused SONIC ONNX and a motion-lib PKL and it
launches a MuJoCo viewer with the policy tracking the clip; ``--no-viewer``
runs the same loop headless, ``--kinematic`` plays the reference without
physics, ``--record`` captures either mode to mp4.

The fused ONNX takes a single 1270-D vector::

    actor_obs = [tokenizer_obs(520) | proprioception(750)]

and returns a 23-D action in IsaacLab DOF order. Observation construction
mirrors the training env exactly (dims from the resolved training config
shipped at ``models/locoft2_final_config.yaml``):

    tokenizer (g1 encoder inputs, per-frame interleaved over 10 future frames):
        command_multi_future_nonflat   (10, 46)  jpos(23)+jvel(23), IL order
        motion_anchor_ori_b_mf_nonflat (10, 6)   6D rot diff vs current base
    proprioception (PolicyCfg attribute order, 10-frame history each,
    oldest-first within each term):
        base_ang_vel(3) | joint_pos_rel(23) | joint_vel(23) |
        last_action(23) | gravity_dir(3)  -> 10 * 75 = 750

    NOTE: the training config's ``joints_idx: [21, 22]`` wrist-future term
    (``joint_pos_multi_future_wrist_for_smpl``) feeds ONLY the ``smpl``
    encoder — the deployed g1 encoder never sees it, so it does not appear
    here.

Asimov quirks handled (see ``gear_sonic/envs/manager_env/robots/asimov.py``):
  - MJCF right side is sign-mirrored vs the left; per-side default pose.
  - MJCF joint order puts the RIGHT arm before the LEFT arm.
  - Hardware-characterized kp/kd/effort used verbatim (no armature-derived
    gains, no X2-style deployment PD scaling); torque is clamped to the
    per-joint effort hard clamp exactly as IsaacLab's ``effort_limit_sim``.
  - Elbow ``ref`` +/-45 deg is baked into the MJCF, and the PKL ``dof``
    convention == MJCF qpos, so joint values pass through untouched.

A tracking summary (survival seconds, mean/max joint |q - q_ref| in rad,
mean |pelvis_z - ref_z|) is printed per episode and aggregated at exit.

Usage (standalone repo — after ./install.sh):

  .venv/bin/python scripts/eval_asimov_mujoco_onnx.py \\
      --onnx models/locoft2_final_9800.onnx \\
      --motion motions/asimov_relaxed_walk.pkl \\
      --clip walk_forward_relax_001__A002

  Headless smoke run:
      ... --no-viewer --total-sim-seconds 20
"""

from __future__ import annotations

import argparse
import collections
import time
from pathlib import Path

import joblib
import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime as ort
from scipy.spatial.transform import Rotation as Rot

# Policy I/O dims (fixed by the training export — see docs/dds_message_schema.md).
NUM_ACTIONS = 23
TOK_DIM = 520          # 10 future frames x (23 jpos + 23 jvel + 6 ori) = 10*52
PROP_DIM = 750         # 10-frame history x (3 angvel + 23 jpos + 23 jvel + 23 action + 3 gravity)
ACTOR_OBS_DIM = TOK_DIM + PROP_DIM  # 1270

REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------- Asimov constants ----------
NUM_DOFS = 23
HISTORY_LEN = 10
CONTROL_DT = 0.02
SIM_DT = 0.005
DECIMATION = 4
NUM_FUTURE_FRAMES = 10
DT_FUTURE_REF = 0.1

MJCF_PATH = str(REPO_ROOT / "assets/mjcf/asimov.xml")

# MuJoCo DFS joint order, from gear_sonic/envs/manager_env/robots/asimov.py
# (ASIMOV_MUJOCO_JOINTS — copied, not imported, because that module pulls in
# isaaclab; asserted against the loaded MJCF at startup instead).
MUJOCO_JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_yaw_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_yaw_joint",
]

# IsaacLab joint order (ASIMOV_ISAACLAB_JOINTS_ORDER in robots/asimov.py,
# runtime-asserted there against robot.joint_names).
ISAACLAB_JOINT_NAMES = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_knee_joint", "right_knee_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_elbow_joint", "right_elbow_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
]

# Generated from the name lists, never hand-typed:
#   arr_il = arr_mj[IL_TO_MJ_DOF]   (gather MJ-ordered values into IL order)
#   arr_mj = arr_il[MJ_TO_IL_DOF]   (gather IL-ordered values into MJ order)
IL_TO_MJ_DOF = np.array(
    [MUJOCO_JOINT_NAMES.index(n) for n in ISAACLAB_JOINT_NAMES], dtype=np.int64
)
MJ_TO_IL_DOF = np.array(
    [ISAACLAB_JOINT_NAMES.index(n) for n in MUJOCO_JOINT_NAMES], dtype=np.int64
)
# Cross-check against ASIMOV_MJ_TO_IL in convert_soma_csv_to_motion_lib.py.
assert MJ_TO_IL_DOF.tolist() == [0, 3, 7, 11, 15, 19, 1, 4, 8, 12, 16, 20, 2,
                                 6, 10, 14, 18, 22, 5, 9, 13, 17, 21]
assert (np.asarray(MJ_TO_IL_DOF)[IL_TO_MJ_DOF] == np.arange(NUM_DOFS)).all()

# Hardware-characterized per-joint parameters: basename -> (kp, kd,
# effort_hard_clamp). Efforts are the actuators' SATURATION (peak) ratings —
# what the policy trained with. MUST match the training config exactly:
# effort feeds both the torque clamp AND the action scale (0.25*effort/kp),
# so a mismatch changes action semantics vs training.
JOINT_PARAMS = {
    "hip_pitch": (150.0, 5.0, 120.0),
    "hip_roll": (150.0, 5.0, 90.0),
    "hip_yaw": (150.0, 5.0, 60.0),
    "knee": (150.0, 5.0, 75.0),
    "ankle_pitch": (440.0, 20.0, 145.4),
    "ankle_roll": (440.0, 20.0, 57.6),
    "waist_yaw": (65.0, 5.0, 120.0),
    "shoulder_pitch": (57.0, 5.0, 90.0),
    "shoulder_roll": (86.0, 5.0, 75.0),
    "shoulder_yaw": (96.0, 5.0, 60.0),
    "elbow": (40.0, 2.0, 36.0),
    "wrist_yaw": (40.0, 2.0, 36.0),
}

# Training reset pose (ASIMOV_CFG InitialStateCfg; right side sign-mirrored).
DEFAULT_JOINT_POS = {
    "left_hip_pitch_joint": 0.1,
    "right_hip_pitch_joint": -0.1,
    "left_shoulder_pitch_joint": -0.35,
    "right_shoulder_pitch_joint": 0.35,
    "left_shoulder_roll_joint": -0.18,
    "right_shoulder_roll_joint": 0.18,
    "left_elbow_joint": 0.87,
    "right_elbow_joint": -0.87,
}

# Action clip from the training wrapper (config action_clip_value: 20.0).
ACTION_CLIP = 20.0

# Local MJCF jnt_axis per MuJoCo-ordered joint (convert_soma_csv_to_motion_lib
# ASIMOV_DOF_AXIS). Right side is sign-mirrored; wrist axis is canted.
DOF_AXIS = np.array(
    [
        [0, 1, 0], [1, 0, 0], [0, 0, -1], [0, 1, 0], [0, 1, 0], [-1, 0, 0],      # left leg
        [0, -1, 0], [1, 0, 0], [0, 0, -1], [0, -1, 0], [0, -1, 0], [-1, 0, 0],  # right leg
        [0, 0, 1],                                                               # waist_yaw
        [0, -1, 0], [-1, 0, 0], [0, 0, -1], [0, 1, 0], [0.766044, 0, -0.642788],  # right arm
        [0, 1, 0], [-1, 0, 0], [0, 0, -1], [0, -1, 0], [0.766044, 0, -0.642788],  # left arm
    ],
    dtype=np.float64,
)


# ---------------------------------------------------------------------------
# "Served" DOF convention (sim2sim-gap root cause, found 2026-07-28)
# ---------------------------------------------------------------------------
# The training motion lib does NOT serve the PKL's `dof` (MuJoCo-qpos
# convention). It recomputes joint angles from `pose_aa` via
# torch_humanoid_batch.rotation_matrices_to_dof: three independent atan2
# extractions (x=atan2(R21,R22), y=atan2(R02,R00), z=atan2(R10,R11)) dotted
# with the joint axis — which lands in the |axis| convention. Net effect vs
# raw qpos: every negative-axis joint (mirrored right side, yaw axes,
# ankle/shoulder rolls) is SIGN-FLIPPED, and the canted wrist axis gets a
# nonlinear remap. Everything REFERENCE-side uses this served convention:
#   * tokenizer command jpos (verified vs IsaacLab capture: 3.4e-4 mean)
#   * tokenizer command jvel = forward-diff of served dof (1.9e-3 mean)
#   * RSI reset joint state = served dof CLAMPED to the 0.9-soft joint
#     limits (IsaacLab write clamps; residual 0.016 = reset noise)
# while the PROPRIOCEPTION stream stays in raw MuJoCo/URDF qpos convention
# (robot.data.joint_pos; deltas vs capture = configured obs noise only).
# The policy was trained with this convention split, so deploy must
# replicate it exactly — feeding raw-qpos references makes the policy
# track a ghost target and collapse to the default pose (the pre-fix
# "conservative gait": joint MAE ~0.29 vs IsaacLab-like ~0.17).

def _served_dof_from_pose_aa(pose_aa: np.ndarray) -> np.ndarray:
    """Motion-lib style DOF extraction, (T, 24, 3) pose_aa -> (T, 23) MJ order."""
    aa = np.asarray(pose_aa, dtype=np.float64)[:, 1 : 1 + NUM_DOFS, :]
    T = aa.shape[0]
    R = Rot.from_rotvec(aa.reshape(-1, 3)).as_matrix().reshape(T, NUM_DOFS, 3, 3)
    x = np.arctan2(R[..., 2, 1], R[..., 2, 2])
    y = np.arctan2(R[..., 0, 2], R[..., 0, 0])
    z = np.arctan2(R[..., 1, 0], R[..., 1, 1])
    angles = np.stack([x, y, z], axis=-1)
    return (angles * np.abs(DOF_AXIS)[None]).sum(-1)


def _ensure_served(m) -> None:
    """Attach served-convention dof/vel streams to a motion dict (cached).

    ``_dof_served``: (T, 23) MJ order, motion-lib convention (see above).
    ``_dof_vel_served``: forward difference * fps, last row repeated —
    matches motion-lib dof_vel (verified 1.9e-3 mean vs capture).
    Falls back to a per-joint sign flip of ``dof`` if ``pose_aa`` is missing
    (exact for all joints except the two canted wrists).
    """
    if "_dof_served" in m:
        return
    if "pose_aa" in m:
        served = _served_dof_from_pose_aa(m["pose_aa"])
    else:
        sign = np.array([a[np.argmax(np.abs(a))] for a in DOF_AXIS])
        served = np.asarray(m["dof"], dtype=np.float64) * np.sign(sign)[None]
    fps = float(m["fps"])
    vel = np.empty_like(served)
    vel[:-1] = (served[1:] - served[:-1]) * fps
    vel[-1] = vel[-2] if len(served) > 1 else 0.0
    m["_dof_served"] = served
    m["_dof_vel_served"] = vel


def _basename(jname: str) -> str:
    return jname.replace("left_", "").replace("right_", "").replace("_joint", "")


def _build_arrays(params=None):
    params = JOINT_PARAMS if params is None else params
    kp = np.zeros(NUM_DOFS)
    kd = np.zeros(NUM_DOFS)
    effort = np.zeros(NUM_DOFS)
    action_scale = np.zeros(NUM_DOFS)
    default_pos = np.zeros(NUM_DOFS)
    for i, jname in enumerate(MUJOCO_JOINT_NAMES):
        p_kp, p_kd, p_eff = params[_basename(jname)]
        kp[i], kd[i], effort[i] = p_kp, p_kd, p_eff
        # Repo convention (ASIMOV_ACTION_SCALE): 0.25 * effort / kp.
        action_scale[i] = 0.25 * p_eff / p_kp
        default_pos[i] = DEFAULT_JOINT_POS.get(jname, 0.0)
    return kp, kd, effort, action_scale, default_pos


KP, KD, EFFORT_LIMIT, ACTION_SCALE, DEFAULT_DOF = _build_arrays()



def _soft_joint_limits():
    """0.9-soft joint limits from the MJCF, MJ order (IsaacLab
    soft_joint_pos_limit_factor=0.9: mid +/- 0.9*half). RSI joint writes are
    clamped to these — verified against the IsaacLab capture (clamped joints
    land exactly on mid +/- 0.9*half: right_knee -0.075, left_elbow 0.122)."""
    model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    rng = np.array([model.joint(n).range.copy() for n in MUJOCO_JOINT_NAMES])
    mid = rng.mean(axis=1)
    half = (rng[:, 1] - rng[:, 0]) / 2.0
    return mid - 0.9 * half, mid + 0.9 * half


SOFT_JPOS_LO, SOFT_JPOS_HI = _soft_joint_limits()


# ---------- Output safety layer (deploy-style, bare minimum) ----------
class SafetyLimiter:
    """Two deploy-style limits on the policy output before it drives the sim:

      * action cap:  |action| <= max_action   (tighter than training's ±20)
      * dev clamp:   |target - default_pose| <= max_dev  per joint (rad)

    Applied downstream of the network only — the last_action history obs
    keeps the training convention (±20 clip), exactly like the robot deploy
    stack. Limits are sized to NEVER engage in nominal tracking; engagements
    are counted and reported so a clean run proves the margins.
    """

    def __init__(self, max_action: float, max_dev: float):
        self.max_action = float(max_action)
        self.max_dev = float(max_dev)
        self.ticks = 0
        self.action_hits = 0
        self.dev_hits = 0
        self.worst_action = 0.0
        self.worst_dev = 0.0

    def cap_action(self, action: np.ndarray) -> np.ndarray:
        self.ticks += 1
        m = float(np.max(np.abs(action)))
        self.worst_action = max(self.worst_action, m)
        if m > self.max_action:
            self.action_hits += 1
        return np.clip(action, -self.max_action, self.max_action)

    def clamp_target(self, target: np.ndarray) -> np.ndarray:
        dev = target - DEFAULT_DOF
        m = float(np.max(np.abs(dev)))
        self.worst_dev = max(self.worst_dev, m)
        if m > self.max_dev:
            self.dev_hits += 1
        return DEFAULT_DOF + np.clip(dev, -self.max_dev, self.max_dev)

    def summary(self) -> str:
        n = max(self.ticks, 1)
        clean = self.action_hits + self.dev_hits == 0
        return "\n".join([
            f"  Action cap  (|a| <= {self.max_action:g}):      "
            f"{self.action_hits} ticks ({100.0 * self.action_hits / n:.2f}%), "
            f"worst |a| {self.worst_action:.3f}",
            f"  Dev clamp   (|q*-q0| <= {self.max_dev:g} rad): "
            f"{self.dev_hits} ticks ({100.0 * self.dev_hits / n:.2f}%), "
            f"worst dev {self.worst_dev:.3f} rad",
            f"  Verdict:    "
            f"{'CLEAN (limits never engaged)' if clean else 'ENGAGED — inspect before deploy'}",
        ])


# ---------- Offscreen video recorder ----------
class VideoRecorder:
    """Offscreen MuJoCo render piped to ffmpeg (H.264 mp4).

    Uses the same tracking-camera framing as the interactive viewer. Needs
    ``ffmpeg`` on PATH; headless GL (set ``MUJOCO_GL=egl`` if there is no
    display). Recording is optional — nothing else depends on it.
    """

    def __init__(self, path: str, model, track_body_id: int, fps: float,
                 width: int = 960, height: int = 720):
        import shutil
        import subprocess
        if shutil.which("ffmpeg") is None:
            raise SystemExit("--record needs ffmpeg on PATH")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.cam.trackbodyid = track_body_id
        self.cam.azimuth, self.cam.elevation, self.cam.distance = 120, -20, 2.5
        self.frames = 0
        self.path = path
        self.proc = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{width}x{height}", "-r", f"{fps}", "-i", "-",
             "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
             path],
            stdin=subprocess.PIPE,
        )

    def capture(self, data) -> None:
        self.renderer.update_scene(data, camera=self.cam)
        self.proc.stdin.write(self.renderer.render().tobytes())
        self.frames += 1

    def close(self) -> None:
        self.proc.stdin.close()
        self.proc.wait()
        print(f"  [record] wrote {self.frames} frames -> {self.path}", flush=True)


# ---------- Actors ----------
class OnnxActor:
    """Thin wrapper around the fused Asimov ONNX session.

    ``tokenizer_obs`` must be in the per-frame interleaved layout produced by
    :func:`build_tokenizer_obs` (cmd_f0(46)|ori_f0(6)|...|cmd_f9|ori_f9 —
    exactly what the fused graph reshapes to (B, 10, 52)).
    """

    def __init__(self, onnx_path: str, providers=None):
        if providers is None:
            providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise RuntimeError(
                f"Expected exactly 1 input and 1 output on the fused ONNX, "
                f"got {len(inputs)} inputs / {len(outputs)} outputs"
            )
        self.input_name = inputs[0].name
        self.output_name = outputs[0].name
        self.input_shape = inputs[0].shape
        self.output_shape = outputs[0].shape
        actual_width = self.input_shape[-1]
        if actual_width != ACTOR_OBS_DIM:
            raise RuntimeError(
                f"ONNX input width {actual_width} != expected {ACTOR_OBS_DIM} "
                f"({TOK_DIM} tokenizer + {PROP_DIM} proprioception). Was this "
                f"ONNX exported for a different robot than Asimov (23 DOF)?"
            )

    def __call__(self, proprioception: np.ndarray, tokenizer_obs: np.ndarray) -> np.ndarray:
        actor_obs = np.concatenate(
            [tokenizer_obs.astype(np.float32), proprioception.astype(np.float32)]
        ).reshape(1, -1)
        out = self.session.run([self.output_name], {self.input_name: actor_obs})[0]
        return out[0]  # (23,) IL order

    def describe(self) -> str:
        return (
            f"input '{self.input_name}' shape={self.input_shape} -> "
            f"output '{self.output_name}' shape={self.output_shape}"
        )


# ---------- Motion helpers ----------
def load_motion_data(path):
    return joblib.load(path)


def _m(data):
    return data[list(data.keys())[0]]


def get_total_frames(data):
    return _m(data)["dof"].shape[0]


def get_motion_fps(data):
    return float(_m(data)["fps"])


def quat_rotate_inverse(q_wxyz, v):
    """Rotate v by the INVERSE of quaternion q (wxyz). Matches IsaacLab."""
    w, x, y, z = q_wxyz
    u = np.array([x, y, z])
    t = 2.0 * np.cross(u, v)
    return v - w * t + np.cross(u, t)


def compute_motion_state(motion_data, frame, fps):
    """Full RSI reset state from a motion frame.

    Joint state uses the SERVED convention clamped to the 0.9-soft limits —
    mirroring IsaacLab's ``write_joint_state_to_sim`` of motion-lib data
    (see the served-DOF note above). Root state comes from the PKL raw
    streams (``root_rot`` xyzw, ``root_trans_offset`` world frame), with
    velocities reconstructed by one-step finite difference.
    """
    m = _m(motion_data)
    _ensure_served(m)
    n_frames = m["dof"].shape[0]
    f = int(frame) % n_frames
    f_next = min(f + 1, n_frames - 1)
    dt = 1.0 / float(fps)

    joint_pos_mj = np.clip(m["_dof_served"][f], SOFT_JPOS_LO, SOFT_JPOS_HI)
    joint_vel_mj = m["_dof_vel_served"][f].copy()

    root_pos_w = np.asarray(m["root_trans_offset"][f], dtype=np.float64).copy()
    quat_xyzw = np.asarray(m["root_rot"][f], dtype=np.float64)
    root_quat_w_wxyz = np.array(
        [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float64
    )

    if f_next != f:
        root_lin_vel_w = (
            np.asarray(m["root_trans_offset"][f_next], dtype=np.float64) - root_pos_w
        ) / dt
        q_next_xyzw = np.asarray(m["root_rot"][f_next], dtype=np.float64)
        rel = Rot.from_quat(q_next_xyzw) * Rot.from_quat(quat_xyzw).inv()
        root_ang_vel_w = rel.as_rotvec() / dt
    else:
        root_lin_vel_w = np.zeros(3, dtype=np.float64)
        root_ang_vel_w = np.zeros(3, dtype=np.float64)

    return {
        "joint_pos_mj": joint_pos_mj,
        "joint_vel_mj": joint_vel_mj,
        "root_pos_w": root_pos_w,
        "root_quat_w_wxyz": root_quat_w_wxyz,
        "root_lin_vel_w": root_lin_vel_w,
        "root_ang_vel_w": root_ang_vel_w,
    }


# ---------- Observation construction ----------
class ProprioceptionBuffer:
    """Per-term history buffers matching IsaacLab's CircularBuffer semantics.

    First append after reset broadcast-fills all HISTORY_LEN slots. Flat
    layout (PolicyCfg attribute order, oldest-first within each term):
    base_ang_vel | joint_pos_rel | joint_vel | last_action | gravity_dir.
    """

    def __init__(self):
        self.gravity_hist = collections.deque(maxlen=HISTORY_LEN)
        self.angvel_hist = collections.deque(maxlen=HISTORY_LEN)
        self.jpos_hist = collections.deque(maxlen=HISTORY_LEN)
        self.jvel_hist = collections.deque(maxlen=HISTORY_LEN)
        self.action_hist = collections.deque(maxlen=HISTORY_LEN)
        self._primed = False

    def reset(self):
        for h in (self.gravity_hist, self.angvel_hist, self.jpos_hist,
                  self.jvel_hist, self.action_hist):
            h.clear()
        self._primed = False

    def append(self, gravity, angvel, jpos_rel, jvel, action):
        vals = [gravity, angvel, jpos_rel, jvel, action]
        hists = [self.gravity_hist, self.angvel_hist, self.jpos_hist,
                 self.jvel_hist, self.action_hist]
        reps = 1 if self._primed else HISTORY_LEN
        for h, v in zip(hists, vals):
            v32 = v.astype(np.float32)
            for _ in range(reps):
                h.append(v32)
        self._primed = True

    def get_flat(self) -> np.ndarray:
        parts = []
        for hist in [self.angvel_hist, self.jpos_hist, self.jvel_hist,
                     self.action_hist, self.gravity_hist]:
            parts.extend(hist)
        flat = np.concatenate(parts).astype(np.float32)
        assert flat.shape[0] == PROP_DIM
        return flat


def build_tokenizer_obs(motion_data, current_time, base_quat_wxyz, motion_fps):
    """Build the 520-D tokenizer input matching IsaacLab's exact layout.

    command_flat = cat([jpos_flat(10*23), jvel_flat(10*23)]).reshape(10, 46)
    ori          = 6D rot diff per future frame, (10, 6), row-major
                   mat[:, :2].reshape(-1) (see eval_x2_mujoco for the
                   column-major bug archaeology)
    encoder_input = cat(command, ori, dim=-1).reshape(-1) = 520

    Future window starts at the CURRENT frame (t + 0.0) then steps by
    DT_FUTURE_REF (TrackingCommand.future_time_steps_init).
    """
    m = _m(motion_data)
    _ensure_served(m)
    total_frames = m["dof"].shape[0]
    dt = 1.0 / motion_fps

    cur_rot = Rot.from_quat([base_quat_wxyz[1], base_quat_wxyz[2],
                             base_quat_wxyz[3], base_quat_wxyz[0]])

    # Future-frame indices in INTEGER step space, mirroring IsaacLab's
    # ``arange(num_future_frames) * frame_skips`` (frame 0 = current frame).
    # An earlier float version (``int((t + f*0.1) * fps)``) hit floor errors
    # (e.g. 4.56*50 -> 227.99999 -> 227), corrupting the served jvel at
    # velocity spikes.
    base_frame = int(round(current_time * motion_fps))
    frame_skip = int(round(DT_FUTURE_REF * motion_fps))
    jpos_frames, jvel_frames, ori_frames = [], [], []
    for f in range(NUM_FUTURE_FRAMES):
        fi = min(base_frame + f * frame_skip, total_frames - 1)

        # Reference joint streams in the SERVED convention (see module note):
        # this is what the policy was trained to consume; raw `dof` here is
        # the sim2sim bug that made the policy track a ghost target.
        jpos_il = m["_dof_served"][fi][IL_TO_MJ_DOF]
        jpos_frames.append(jpos_il.astype(np.float32))
        jvel_frames.append(m["_dof_vel_served"][fi][IL_TO_MJ_DOF].astype(np.float32))

        fq = m["root_rot"][fi]  # xyzw
        relative = cur_rot.inv() * Rot.from_quat(fq)
        ori_frames.append(relative.as_matrix()[:, :2].reshape(-1).astype(np.float32))

    command_flat = np.concatenate([np.concatenate(jpos_frames),
                                   np.concatenate(jvel_frames)])
    command_nonflat = command_flat.reshape(NUM_FUTURE_FRAMES, -1)   # (10, 46)
    ori_nonflat = np.stack(ori_frames)                              # (10, 6)
    encoder_input = np.concatenate([command_nonflat, ori_nonflat], axis=-1)
    flat = encoder_input.reshape(-1).astype(np.float32)             # (520,)
    assert flat.shape[0] == TOK_DIM
    return flat


# ---------- Tracking metrics ----------
class TrackingMeter:
    """Per-episode joint/root tracking error accumulator (mpjpe-ish, in rad/m)."""

    def __init__(self):
        self.episodes: list[dict] = []
        self._reset_accum()

    def _reset_accum(self):
        self._jmae, self._jmax, self._zerr, self._n = 0.0, 0.0, 0.0, 0

    def step(self, qpos_mj, ref_dof_mj, pelvis_z, ref_z):
        err = np.abs(qpos_mj - ref_dof_mj)
        self._jmae += float(err.mean())
        self._jmax = max(self._jmax, float(err.max()))
        self._zerr += abs(float(pelvis_z) - float(ref_z))
        self._n += 1

    @staticmethod
    def classify(reason: str) -> str:
        """motion_end = clean clip completion; truncated = harness cut the
        episode short (budget/limit/manual), NOT a policy failure; fall =
        pelvis-height or tilt termination."""
        if reason == "motion_end":
            return "motion_end"
        if reason in ("time_budget", "manual") or reason.startswith("reached --max-episode"):
            return "truncated"
        return "fall"

    def end_episode(self, seconds: float, reason: str) -> str:
        n = max(self._n, 1)
        kind = self.classify(reason)
        ep = {
            "seconds": seconds,
            "joint_mae": self._jmae / n,
            "joint_maxerr": self._jmax,
            "z_mae": self._zerr / n,
            "reason": reason,
            "kind": kind,
        }
        self.episodes.append(ep)
        self._reset_accum()
        note = " (truncated by harness, not a fall)" if kind == "truncated" else ""
        return (f"survived {seconds:.2f}s | joint MAE {ep['joint_mae']:.4f} rad "
                f"(max {ep['joint_maxerr']:.3f}) | pelvis-z MAE {ep['z_mae']:.3f} m "
                f"| end: {reason}{note}")

    def summary(self) -> str:
        if not self.episodes:
            return "  No completed episodes."
        secs = [e["seconds"] for e in self.episodes]
        maes = [e["joint_mae"] for e in self.episodes]
        kinds = [e["kind"] for e in self.episodes]
        return "\n".join([
            f"  Episodes:            {len(self.episodes)}",
            f"  End reasons:         {kinds.count('motion_end')} motion_end (clean), "
            f"{kinds.count('fall')} fall, {kinds.count('truncated')} truncated by harness",
            f"  Mean survival:       {np.mean(secs):.2f} s (min {np.min(secs):.2f}, "
            f"max {np.max(secs):.2f})",
            f"  Mean joint MAE:      {np.mean(maes):.4f} rad",
            f"  Worst joint err:     {max(e['joint_maxerr'] for e in self.episodes):.3f} rad",
            f"  Mean pelvis-z MAE:   {np.mean([e['z_mae'] for e in self.episodes]):.3f} m",
        ])


# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--onnx", default=None,
                        help="Fused Asimov SONIC ONNX (models/*.onnx). "
                             "Required unless --kinematic.")
    parser.add_argument("--kinematic", action="store_true",
                        help="Kinematic reference playback: pose the robot "
                             "directly from the clip frames (no physics, no "
                             "policy). Viewer, or offscreen with --record.")
    parser.add_argument("--record", default=None, metavar="OUT.mp4",
                        help="Record an offscreen video (implies no realtime "
                             "pacing; combine with --no-viewer). Policy mode "
                             "records at 25 fps, kinematic at the clip fps. "
                             "Needs ffmpeg on PATH.")
    parser.add_argument("--motion", required=True, help="Reference motion PKL.")
    parser.add_argument("--clip", default=None,
                        help="Only play clips whose name CONTAINS this substring "
                             "(case-insensitive). Default: all clips in the PKL, "
                             "iterated one-by-one with an RSI reset before each.")
    parser.add_argument("--push-force", type=float, default=80.0,
                        help="Manual push magnitude (N), keys 1/2/3/4 = "
                             "front/back/left/right at the torso.")
    parser.add_argument("--push-duration", type=float, default=0.25,
                        help="Seconds the push force is held.")
    parser.add_argument("--show-forces", action="store_true",
                        help="Viewer: draw contact points and contact-force "
                             "arrows (direction + magnitude).")
    parser.add_argument("--loop-clips", action="store_true",
                        help="With a multi-clip PKL, cycle back to the first clip "
                             "after the last (default: stop after one pass through "
                             "all clips, matching eval_x2_mujoco --motions). A "
                             "single-clip run always loops. NOTE: without this "
                             "flag a multi-clip run needs no --total-sim-seconds; "
                             "combining looping with a time budget truncates "
                             "whichever episode the budget lands in (reported as "
                             "'time_budget', not a fall).")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--init-frame", type=int, default=0,
                        help="Motion frame to RSI-initialize at (default 0).")
    parser.add_argument("--fall-height", type=float, default=0.35,
                        help="Pelvis z below this (m) triggers a reset "
                             "(default 0.35; Asimov stands at ~0.63).")
    parser.add_argument("--fall-tilt-cos", type=float, default=-0.3,
                        help="gravity_body[z] above this triggers a reset "
                             "(default -0.3 ~ 72 deg tilt).")
    parser.add_argument("--max-episode", type=float, default=0.0,
                        help="If > 0, force-reset after this many sim seconds per episode.")
    parser.add_argument("--total-sim-seconds", type=float, default=0.0,
                        help="If > 0 (headless), exit once cumulative sim seconds "
                             "across all episodes reach this.")
    parser.add_argument("--no-viewer", action="store_true",
                        help="Headless mode: no MuJoCo viewer, no real-time pacing.")
    parser.add_argument("--kp-scale", type=float, default=1.0,
                        help="Global multiplier on deployed KP (A/B only; default 1.0 "
                             "= hardware-characterized gains verbatim).")
    parser.add_argument("--kd-scale", type=float, default=1.0,
                        help="Global multiplier on deployed KD.")
    parser.add_argument("--max-action", type=float, default=12.0,
                        help="Safety cap on |action| (deploy-style, applied "
                             "downstream of the network; training clip is 20). "
                             "0 disables. Default 12.")
    parser.add_argument("--max-dev", type=float, default=2.2,
                        help="Safety clamp on |target - default_pose| per "
                             "joint, rad. 0 disables. Default 2.2.")
    args = parser.parse_args()

    kp = KP * float(args.kp_scale)
    kd = KD * float(args.kd_scale)

    safety = None
    if args.max_action > 0 or args.max_dev > 0:
        safety = SafetyLimiter(
            args.max_action if args.max_action > 0 else float("inf"),
            args.max_dev if args.max_dev > 0 else float("inf"),
        )
        print(f"Safety limits: |action| <= {safety.max_action:g}, "
              f"|target - default| <= {safety.max_dev:g} rad", flush=True)

    if args.record and not args.no_viewer:
        parser.error("--record requires --no-viewer (offscreen render)")
    if args.kinematic and args.no_viewer and not args.record:
        parser.error("--kinematic needs the viewer (or --record for offscreen)")
    if not args.kinematic and args.onnx is None:
        parser.error("--onnx is required unless --kinematic")

    onnx_actor = None
    if not args.kinematic:
        print(f"Loading ONNX session from {args.onnx} ...", flush=True)
        onnx_actor = OnnxActor(args.onnx)
        print(f"  ONNX: {onnx_actor.describe()}", flush=True)

    print(f"Loading motion from {args.motion} ...", flush=True)
    _all = load_motion_data(args.motion)
    clip_names = list(_all.keys())
    if args.clip:
        if args.clip in clip_names:  # exact key match wins over substring
            clip_names = [args.clip]
        else:
            clip_names = [k for k in clip_names if args.clip.lower() in k.lower()]
        if not clip_names:
            raise SystemExit(f"--clip '{args.clip}' matched no clips in {args.motion}")
    clips = [{k: _all[k]} for k in clip_names]
    n_clips = len(clips)
    print(f"  {n_clips} clip(s): {', '.join(clip_names)}", flush=True)

    print("Loading MuJoCo model ...", flush=True)
    mj_model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    mj_data = mujoco.MjData(mj_model)
    mj_model.opt.timestep = SIM_DT

    # Assert the baked joint list matches the MJCF (order and names), and
    # resolve actuator ids by joint so a reordered actuator block can't
    # silently scramble torques.
    mjcf_joints = [mj_model.joint(j).name for j in range(mj_model.njnt)
                   if mj_model.joint(j).type != mujoco.mjtJoint.mjJNT_FREE]
    assert mjcf_joints == MUJOCO_JOINT_NAMES, (
        f"MJCF joint order changed:\n{mjcf_joints}\nvs baked\n{MUJOCO_JOINT_NAMES}")
    joint_to_actuator = np.array(
        [mj_model.actuator(n).id for n in MUJOCO_JOINT_NAMES], dtype=np.int64
    )

    pelvis_id = mj_model.body("pelvis").id
    # Manual-push target: Asimov's neck/head are fused into torso_link, so a
    # "head push" is a torso push (highest available body — loads the waist
    # like eval_x2_mujoco's head_pitch_link default loads X2's waist pitch).
    push_body_id = mj_model.body("torso_link").id
    push_state = {"dir": (1.0, 0.0), "steps_left": 0}

    # ---- Per-clip state ----
    init_frame = int(args.init_frame)
    clip_idx = 0
    motion_data = None
    total_frames = 0
    motion_fps = 30.0
    init_motion_state = None
    init_root_z = 0.63

    def _load_clip(idx):
        nonlocal motion_data, total_frames, motion_fps, init_motion_state, init_root_z
        motion_data = clips[idx]
        total_frames = get_total_frames(motion_data)
        motion_fps = get_motion_fps(motion_data)
        init_motion_state = compute_motion_state(motion_data, init_frame, motion_fps)
        init_root_z = float(init_motion_state["root_pos_w"][2])
        print(f"\n=== CLIP {idx + 1}/{n_clips}: {clip_names[idx]} "
              f"({total_frames} frames @ {motion_fps:g} fps = "
              f"{total_frames / motion_fps:.1f}s) ===", flush=True)

    _load_clip(0)

    recorder = None
    if args.record:
        rec_fps = motion_fps if args.kinematic else 25.0
        recorder = VideoRecorder(args.record, mj_model, pelvis_id, rec_fps)
        print(f"Recording -> {args.record} @ {rec_fps:g} fps", flush=True)

    # ---- Kinematic playback mode: pose from the clip, no physics/policy ----
    if args.kinematic:
        kin = {"paused": False, "frame": float(init_frame), "played": 0,
               "clip": 0}

        def _kin_pose(f: int) -> None:
            m = _m(motion_data)
            mj_data.qpos[0:3] = np.asarray(m["root_trans_offset"][f])
            q = np.asarray(m["root_rot"][f])  # xyzw
            mj_data.qpos[3:7] = [q[3], q[0], q[1], q[2]]
            # `dof` is in the MJCF qpos convention — passes through untouched.
            mj_data.qpos[7:7 + NUM_DOFS] = np.asarray(m["dof"][f])
            mj_data.qvel[:] = 0
            mujoco.mj_forward(mj_model, mj_data)

        def _kin_next_clip() -> None:
            kin["clip"] = (kin["clip"] + 1) % n_clips
            _load_clip(kin["clip"])
            kin["frame"] = float(init_frame)

        def kin_key_callback(keycode):
            import glfw
            if keycode == glfw.KEY_SPACE:
                kin["paused"] = not kin["paused"]
                print("Paused" if kin["paused"] else "Resumed", flush=True)
            elif keycode == glfw.KEY_R:
                kin["frame"] = float(init_frame)
            elif keycode == glfw.KEY_N:
                _kin_next_clip()
            elif keycode == glfw.KEY_V:
                if viewer.cam.type == mujoco.mjtCamera.mjCAMERA_TRACKING:
                    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                else:
                    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                    viewer.cam.trackbodyid = pelvis_id

        print("\n=== Kinematic reference playback (no physics, no policy) ===",
              flush=True)

        if recorder is not None:
            # Offscreen: one pass over the selected clips at clip fps.
            for ci in range(n_clips):
                if ci > 0:
                    _kin_next_clip()
                for f in range(init_frame, total_frames):
                    _kin_pose(f)
                    recorder.capture(mj_data)
            recorder.close()
            return

        print("Press SPACE pause, R restart clip, N next clip, V toggle camera.\n",
              flush=True)
        with mujoco.viewer.launch_passive(
            mj_model, mj_data,
            key_callback=kin_key_callback,
            show_left_ui=False, show_right_ui=False,
        ) as viewer:
            viewer.cam.azimuth = 120
            viewer.cam.elevation = -20
            viewer.cam.distance = 2.5
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = pelvis_id
            while viewer.is_running():
                if kin["paused"]:
                    viewer.sync()
                    time.sleep(0.02)
                    continue
                _kin_pose(int(kin["frame"]) % total_frames)
                viewer.sync()
                time.sleep(1.0 / (motion_fps * max(args.speed, 1e-6)))
                kin["frame"] += 1.0
                if kin["frame"] >= total_frames:
                    kin["played"] += 1
                    if n_clips > 1:
                        if not args.loop_clips and kin["played"] >= n_clips:
                            print(f"  [end] all {n_clips} clips played once "
                                  f"(pass --loop-clips to cycle), exiting.",
                                  flush=True)
                            break
                        _kin_next_clip()
                    else:
                        kin["frame"] = float(init_frame)
        print("Viewer closed.")
        return

    prop_buf = ProprioceptionBuffer()
    meter = TrackingMeter()
    last_action_mj = np.zeros(NUM_DOFS, dtype=np.float32)
    sim_time = float(init_frame) / motion_fps
    step_count = 0
    episode_count = 0
    episode_start_step = 0
    paused = False

    def _apply_init_state():
        s = init_motion_state
        mj_data.qpos[0] = 0.0
        mj_data.qpos[1] = 0.0
        mj_data.qpos[2] = float(s["root_pos_w"][2])
        mj_data.qpos[3:7] = s["root_quat_w_wxyz"]
        mj_data.qpos[7:7 + NUM_DOFS] = s["joint_pos_mj"]
        # MuJoCo free joint: qvel[0:3] world linear, qvel[3:6] BODY angular.
        mj_data.qvel[0:3] = s["root_lin_vel_w"]
        mj_data.qvel[3:6] = quat_rotate_inverse(
            s["root_quat_w_wxyz"], s["root_ang_vel_w"]
        )
        mj_data.qvel[6:6 + NUM_DOFS] = s["joint_vel_mj"]
        mj_data.xfrc_applied[:] = 0
        mujoco.mj_forward(mj_model, mj_data)

    _apply_init_state()

    # Reference-index tracker: a tracking policy can't chase a teleport, so a
    # looping reference frame => full reset, not a silent wrap (see
    # eval_x2_mujoco_onnx for the rationale).
    prev_motion_frame = -1

    # Number of clip-episodes finished so far. A multi-clip run stops after
    # one full pass (clips_played == n_clips) unless --loop-clips: cycling
    # forever means any fixed --total-sim-seconds budget chops a partial
    # re-run of clip 0 whose "survival time" is just the leftover budget --
    # deterministic across checkpoints and easy to misread as a fall.
    clips_played = 0

    def reset_state(reason: str = "", advance_clip: bool = True) -> None:
        nonlocal sim_time, last_action_mj, episode_count, episode_start_step
        nonlocal prev_motion_frame, clip_idx, clips_played
        ep_seconds = (step_count - episode_start_step) * CONTROL_DT
        if step_count > episode_start_step:
            print(f"  [episode {episode_count}] "
                  f"{meter.end_episode(ep_seconds, reason)}", flush=True)
            if advance_clip:
                clips_played += 1
        if advance_clip and n_clips > 1:
            clip_idx = (clip_idx + 1) % n_clips
            _load_clip(clip_idx)
        prev_motion_frame = -1
        sim_time = float(init_frame) / motion_fps
        last_action_mj[:] = 0
        prop_buf.reset()
        _apply_init_state()
        episode_count += 1
        episode_start_step = step_count

    def step_once() -> str | None:
        """Run one control tick. Returns reset reason if the episode ended."""
        nonlocal sim_time, step_count, last_action_mj, prev_motion_frame

        motion_time = sim_time * args.speed
        motion_frame = int(motion_time * motion_fps) % total_frames
        if 0 <= prev_motion_frame and motion_frame < prev_motion_frame:
            return "motion_end"
        prev_motion_frame = motion_frame
        motion_time = motion_frame / motion_fps

        qpos_j = mj_data.qpos[7:7 + NUM_DOFS].copy()
        qvel_j = mj_data.qvel[6:6 + NUM_DOFS].copy()
        base_quat = mj_data.qpos[3:7].copy()
        base_angvel = mj_data.qvel[3:6].copy()

        dof_pos_il = qpos_j[IL_TO_MJ_DOF]
        dof_vel_il = qvel_j[IL_TO_MJ_DOF]
        action_il = last_action_mj[IL_TO_MJ_DOF]

        gravity = quat_rotate_inverse(base_quat, np.array([0.0, 0.0, -1.0]))
        dof_pos_rel_il = dof_pos_il - DEFAULT_DOF[IL_TO_MJ_DOF]

        prop_buf.append(gravity, base_angvel, dof_pos_rel_il, dof_vel_il, action_il)
        proprioception = prop_buf.get_flat()
        tokenizer_obs = build_tokenizer_obs(motion_data, motion_time, base_quat,
                                            motion_fps)

        action_il_drive = onnx_actor(proprioception, tokenizer_obs)
        # Training wrapper clips actions (action_clip_value=20.0); the clipped
        # action is also what enters the last_action history obs.
        action_il_drive = np.clip(action_il_drive, -ACTION_CLIP, ACTION_CLIP)

        action_mj = action_il_drive[MJ_TO_IL_DOF]
        last_action_mj = action_mj.astype(np.float32).copy()
        # Deploy-style safety limits (downstream of the obs history on
        # purpose — the network keeps seeing the training-convention action).
        action_safe = safety.cap_action(action_mj) if safety else action_mj
        target_pos = DEFAULT_DOF + action_safe * ACTION_SCALE
        if safety:
            target_pos = safety.clamp_target(target_pos)

        # Manual push (keys 1-4): world-frame force rotated by current pelvis
        # heading so "front" is always robot-forward. Applied at the torso
        # (Asimov's neck is fused — the torso IS the head-bearing body); auto
        # clears after --push-duration.
        nonlocal_push = push_state
        if nonlocal_push["steps_left"] > 0:
            yaw = Rot.from_quat(mj_data.qpos[3:7][[1, 2, 3, 0]]).as_euler("zyx")[0]
            c, s = np.cos(yaw), np.sin(yaw)
            fx, fy = nonlocal_push["dir"]
            world = np.array([c * fx - s * fy, s * fx + c * fy, 0.0])
            mj_data.xfrc_applied[push_body_id, :3] = world * args.push_force
            nonlocal_push["steps_left"] -= 1
            if nonlocal_push["steps_left"] == 0:
                mj_data.xfrc_applied[push_body_id, :] = 0.0
                print(f"  [push] released", flush=True)

        for _ in range(DECIMATION):
            torque = (
                kp * (target_pos - mj_data.qpos[7:7 + NUM_DOFS])
                - kd * mj_data.qvel[6:6 + NUM_DOFS]
            )
            # Effort hard clamp, exactly as IsaacLab's effort_limit_sim.
            torque = np.clip(torque, -EFFORT_LIMIT, EFFORT_LIMIT)
            mj_data.ctrl[joint_to_actuator] = torque
            mujoco.mj_step(mj_model, mj_data)

        sim_time += CONTROL_DT
        step_count += 1

        pelvis_z = float(mj_data.qpos[2])
        ref_dof = np.asarray(_m(motion_data)["dof"][motion_frame], dtype=np.float64)
        ref_z = float(_m(motion_data)["root_trans_offset"][motion_frame][2])
        meter.step(mj_data.qpos[7:7 + NUM_DOFS], ref_dof, pelvis_z, ref_z)

        grav_z = float(gravity[2])
        episode_seconds = (step_count - episode_start_step) * CONTROL_DT
        if pelvis_z < args.fall_height:
            return f"pelvis_z={pelvis_z:.3f} < {args.fall_height:.2f}"
        if grav_z > args.fall_tilt_cos:
            tilt_deg = int(np.rad2deg(np.arccos(np.clip(-grav_z, -1, 1))))
            return (f"gravity_body[z]={grav_z:+.2f} > {args.fall_tilt_cos:.2f} "
                    f"(tilt {tilt_deg} deg)")
        if args.max_episode > 0 and episode_seconds >= args.max_episode:
            return f"reached --max-episode={args.max_episode:.1f}s"
        if step_count % 250 == 0:
            print(f"[ep {episode_count}] step={step_count} sim={sim_time:.2f}s "
                  f"frame={motion_frame}/{total_frames} h={pelvis_z:.3f}m",
                  flush=True)
        return None

    print("\n=== Asimov MuJoCo Eval (ONNX) ===", flush=True)
    print(f"Robot RSI-initialized from motion frame {init_frame}.", flush=True)
    print(f"Auto-reset triggers: pelvis_z < {args.fall_height:.2f} m, "
          f"or gravity_body[z] > {args.fall_tilt_cos:.2f}.", flush=True)
    if not args.no_viewer:
        print("Press SPACE pause, R reset, N next clip, V toggle camera.\n",
              flush=True)
    else:
        print("Headless mode (no viewer).\n", flush=True)

    headless_exit_after_one_episode = (
        args.no_viewer and args.max_episode > 0 and args.total_sim_seconds <= 0
    )
    exit_requested = False
    cumulative_sim_seconds = 0.0

    def _pass_complete() -> bool:
        return n_clips > 1 and not args.loop_clips and clips_played >= n_clips

    if args.no_viewer:
        while not exit_requested:
            reason = step_once()
            # A "motion_end" tick returns before stepping physics, so it
            # consumes no sim time (keeps --total-sim-seconds accounting exact).
            if reason != "motion_end":
                cumulative_sim_seconds += CONTROL_DT
                # 50 Hz control -> capture every 2nd tick = 25 fps video.
                if recorder is not None and step_count % 2 == 0:
                    recorder.capture(mj_data)
            if (args.total_sim_seconds > 0
                    and cumulative_sim_seconds >= args.total_sim_seconds):
                print(f"  [end] cumulative sim time {cumulative_sim_seconds:.2f}s "
                      f">= --total-sim-seconds={args.total_sim_seconds:.1f}s, "
                      f"exiting.", flush=True)
                reset_state("time_budget", advance_clip=False)  # flush metrics
                exit_requested = True
                continue
            if reason is not None:
                if headless_exit_after_one_episode:
                    reset_state(reason, advance_clip=False)
                    exit_requested = True
                else:
                    reset_state(reason)
                    if _pass_complete():
                        print(f"  [end] all {n_clips} clips played once "
                              f"(pass --loop-clips to cycle), exiting.", flush=True)
                        exit_requested = True
        if recorder is not None:
            recorder.close()
    else:
        def key_callback(keycode):
            nonlocal paused
            import glfw
            if keycode == glfw.KEY_SPACE:
                paused = not paused
                print("Paused" if paused else "Resumed", flush=True)
            elif keycode == glfw.KEY_R:
                reset_state("manual", advance_clip=False)
            elif keycode == glfw.KEY_N:
                reset_state("manual_next_clip")
            elif keycode in (glfw.KEY_UP, glfw.KEY_DOWN, glfw.KEY_LEFT, glfw.KEY_RIGHT):
                # Arrow keys: MuJoCo viewer built-ins don't use them (digits
                # toggle geom groups — collision found the hard way).
                dirs = {glfw.KEY_UP: (1.0, 0.0, "front"), glfw.KEY_DOWN: (-1.0, 0.0, "back"),
                        glfw.KEY_LEFT: (0.0, 1.0, "left"), glfw.KEY_RIGHT: (0.0, -1.0, "right")}
                fx, fy, name = dirs[keycode]
                push_state["dir"] = (fx, fy)
                push_state["steps_left"] = max(1, int(args.push_duration / CONTROL_DT))
                print(f"  [push] {name} {args.push_force:.0f} N for "
                      f"{args.push_duration:.2f}s", flush=True)
            elif keycode == glfw.KEY_V:
                if viewer.cam.type == mujoco.mjtCamera.mjCAMERA_TRACKING:
                    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                else:
                    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                    viewer.cam.trackbodyid = pelvis_id

        push_state["steps_left"] = 0  # reset on viewer start

        with mujoco.viewer.launch_passive(
            mj_model, mj_data,
            key_callback=key_callback,
            show_left_ui=False, show_right_ui=False,
        ) as viewer:
            viewer.cam.azimuth = 120
            viewer.cam.elevation = -20
            viewer.cam.distance = 2.5
            viewer.cam.lookat[:] = [0.0, 0.0, init_root_z]
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = pelvis_id
            if args.show_forces:
                # Contact points + force arrows (direction = arrow, magnitude
                # = length). vis.map.force scales meters-per-Newton; the
                # default is tuned for tiny models — shrink so a ~350 N
                # heel-strike stays readable at robot scale.
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = True
                mj_model.vis.map.force = 0.005
                mj_model.vis.scale.contactwidth = 0.05
                mj_model.vis.scale.contactheight = 0.02
                mj_model.vis.scale.forcewidth = 0.02

            wall_start = time.time() - sim_time
            while viewer.is_running():
                if paused:
                    viewer.sync()
                    time.sleep(0.02)
                    continue
                reason = step_once()
                viewer.sync()
                wall_elapsed = time.time() - wall_start
                if sim_time > wall_elapsed:
                    time.sleep(sim_time - wall_elapsed)
                if reason is not None:
                    reset_state(reason)
                    if _pass_complete():
                        print(f"  [end] all {n_clips} clips played once "
                              f"(pass --loop-clips to cycle), exiting.", flush=True)
                        break
                    wall_start = time.time() - sim_time
        print("Viewer closed.")

    print("\n=== Tracking summary ===", flush=True)
    print(meter.summary(), flush=True)

    if safety is not None:
        print("\n=== Safety limits ===", flush=True)
        print(safety.summary(), flush=True)


if __name__ == "__main__":
    main()
