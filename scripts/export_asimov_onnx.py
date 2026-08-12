#!/usr/bin/env python3
"""Export the fused Asimov SONIC deploy ONNX straight from a .pt checkpoint.

Produces the same fused-graph contract as the X2 deploy ONNX
(``reexport_x2_g1_onnx.py``): a single flat input

    actor_obs = [tokenizer_obs(520) | proprioception(750)]  -> (B, 1270)

through g1 encoder -> FSQ -> g1_dyn decoder, returning a (B, 23) action mean
in IsaacLab DOF order.

Asimov dims (from the resolved training config next to the checkpoint,
``sonic_asimov_loco`` / robot.type=asimov, 23 DOF):

    command_multi_future_nonflat:   (10, 46)   10 future frames x (23 jpos + 23 jvel)
    motion_anchor_ori_b_mf_nonflat: (10, 6)    6D rot diff per frame
    tokenizer width:                10 * 52 = 520
    proprioception (actor_obs):     10 * (3 ang_vel + 23 jpos + 23 jvel
                                          + 23 action + 3 gravity) = 750
    encoder: MLP [520, 2048, 1024, 512, 512, 64], SiLU
    FSQ:     32 levels, max_num_tokens=2, token_dim=32 (no params)
    decoder: MLP [64+750=814, 2048, 2048, 1024, 1024, 512, 512, 23], SiLU

Unlike ``reexport_x2_g1_onnx.py`` this does NOT boot an IsaacLab eval to
capture the live ``UniversalTokenModule`` — it rebuilds the exact inference
path from the checkpoint's ``policy_state_dict`` (the same reimplementation
approach as ``eval_x2_mujoco.UniversalTokenActor``, which was verified
against the live module to ~3.6e-7 rad on X2; the architecture is identical
here, only the dims differ). This keeps the export runnable while another
IsaacLab instance owns the GPU. All layer dims are read from the state-dict
tensor shapes, so a mismatch explodes loudly at load time.

Validation: the exported ONNX is compared against the PyTorch fused module
on a batch of random structured inputs; the file is only promoted to
``--output`` if max|onnx - pt| < ``--max-action-diff`` (default 1e-3 rad;
observed ~1e-6).

Usage (local, env_isaaclab python — NOT ``conda activate``):

  ~/miniconda3/envs/env_isaaclab/bin/python \\
      gear_sonic/scripts/export_asimov_onnx.py \\
      --checkpoint out/asimov_bigrun_evals/last_2344.pt \\
      --output out/asimov_bigrun_evals/last_2344_asimov.onnx
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

# ---------------------------------------------------------------------------
# Asimov policy dims (see module docstring for the derivation)
# ---------------------------------------------------------------------------
NUM_ACTIONS = 23
TOK_DIM = 520
PROP_DIM = 750
ACTOR_OBS_DIM = TOK_DIM + PROP_DIM  # 1270
MAX_NUM_TOKENS = 2
TOKEN_DIM = 32
FSQ_LEVELS = 32
ENCODER_NAME = "g1"
DECODER_NAME = "g1_dyn"


def tolerant_torch_load(path: str):
    """torch.load that stubs out unimportable classes (HF-trl etc.).

    Same trick as ``eval_x2_mujoco.load_actor_from_checkpoint``: we only need
    ``policy_state_dict`` (plain tensors); optimizer/args/env objects become
    inert stubs.
    """
    import pickle as _pickle
    import types as _types

    class _Stub:
        def __init__(self, *a, **k):
            pass

        def __setstate__(self, state):
            pass

    class _StubUnpickler(_pickle.Unpickler):
        def find_class(self, module, name):
            try:
                return super().find_class(module, name)
            except Exception:
                return _Stub

    _tolerant = _types.ModuleType("_tolerant_pickle")
    for _a in dir(_pickle):
        setattr(_tolerant, _a, getattr(_pickle, _a))
    _tolerant.Unpickler = _StubUnpickler
    return torch.load(path, map_location="cpu", weights_only=False,
                      pickle_module=_tolerant)


def _mlp_from_state_dict(sd: dict, prefix: str) -> tuple[nn.Sequential, list[int]]:
    """Rebuild a SimpleMLP (Linear/SiLU stack) from ``{prefix}{i}.weight`` keys.

    Layer dims come from the tensor shapes, so nothing is hard-coded.
    """
    idxs = sorted(
        int(k[len(prefix):].split(".")[0])
        for k in sd if k.startswith(prefix) and k.endswith(".weight")
    )
    if not idxs:
        raise KeyError(f"no keys with prefix '{prefix}' in state dict")
    layers: list[nn.Module] = []
    dims = [sd[f"{prefix}{idxs[0]}.weight"].shape[1]]
    for pos, i in enumerate(idxs):
        w = sd[f"{prefix}{i}.weight"]
        lin = nn.Linear(w.shape[1], w.shape[0])
        with torch.no_grad():
            lin.weight.copy_(w)
            lin.bias.copy_(sd[f"{prefix}{i}.bias"])
        layers.append(lin)
        dims.append(w.shape[0])
        if pos < len(idxs) - 1:
            layers.append(nn.SiLU())
    return nn.Sequential(*layers), dims


class FsqQuantizer(nn.Module):
    """vector_quantize_pytorch.FSQ with uniform ``levels`` per dim (no params).

    Matches ``eval_x2_mujoco.fsq_quantize`` exactly:
        half_l  = (L-1) * (1+eps) / 2
        offset  = 0.5 if L even else 0
        shift   = atanh(offset / half_l)
        bounded = tanh(z + shift) * half_l - offset
        out     = round(bounded) / (L // 2)
    """

    def __init__(self, levels: int = FSQ_LEVELS):
        super().__init__()
        L = float(levels)
        eps = 1e-3
        self.half_l = (L - 1.0) * (1.0 + eps) / 2.0
        self.offset = 0.5 if int(levels) % 2 == 0 else 0.0
        self.shift = math.atanh(self.offset / self.half_l) if self.offset else 0.0
        self.div = int(levels) // 2

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        bounded = torch.tanh(z + self.shift) * self.half_l - self.offset
        return torch.round(bounded) / self.div


class FusedAsimovWrapper(nn.Module):
    """Fused encoder + FSQ + decoder, flat (B, 1270) in -> (B, 23) out."""

    def __init__(self, encoder: nn.Sequential, decoder: nn.Sequential):
        super().__init__()
        self.encoder = encoder
        self.quantizer = FsqQuantizer()
        self.decoder = decoder

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        tok = obs[:, :TOK_DIM]
        prop = obs[:, TOK_DIM:]
        latent = self.encoder(tok)                                  # (B, 64)
        latent = latent.view(-1, MAX_NUM_TOKENS, TOKEN_DIM)          # (B, 2, 32)
        quantized = self.quantizer(latent)
        token_flat = quantized.reshape(-1, MAX_NUM_TOKENS * TOKEN_DIM)
        return self.decoder(torch.cat([token_flat, prop], dim=-1))   # (B, 23)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", required=True,
                        help=".pt training checkpoint (policy_state_dict).")
    parser.add_argument("--output", required=True,
                        help="Where to write the fused ONNX.")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--max-action-diff", type=float, default=1e-3,
                        help="Refuse to promote the ONNX if max|onnx-pt| over "
                             "the validation batch exceeds this (radians).")
    parser.add_argument("--val-batch", type=int, default=256,
                        help="Validation batch size of random structured inputs.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true",
                        help="Write the ONNX even if validation fails.")
    args = parser.parse_args()

    print(f"[export_asimov_onnx] Loading checkpoint {args.checkpoint} ...", flush=True)
    ckpt = tolerant_torch_load(args.checkpoint)
    sd = ckpt.get("policy_state_dict") or ckpt.get("actor_model_state_dict")
    if sd is None:
        raise KeyError("Cannot find policy state dict in checkpoint "
                       f"(keys: {list(ckpt.keys())})")

    enc_prefix = f"actor_module.encoders.{ENCODER_NAME}.module."
    dec_prefix = f"actor_module.decoders.{DECODER_NAME}.module."
    encoder, enc_dims = _mlp_from_state_dict(sd, enc_prefix)
    decoder, dec_dims = _mlp_from_state_dict(sd, dec_prefix)
    print(f"[export_asimov_onnx] encoder dims: {enc_dims}", flush=True)
    print(f"[export_asimov_onnx] decoder dims: {dec_dims}", flush=True)

    # Hard shape gates — a wrong-robot checkpoint must die here, not at deploy.
    if enc_dims[0] != TOK_DIM:
        raise ValueError(f"encoder input {enc_dims[0]} != tokenizer width {TOK_DIM}")
    if enc_dims[-1] != MAX_NUM_TOKENS * TOKEN_DIM:
        raise ValueError(f"encoder output {enc_dims[-1]} != "
                         f"{MAX_NUM_TOKENS}*{TOKEN_DIM}")
    if dec_dims[0] != MAX_NUM_TOKENS * TOKEN_DIM + PROP_DIM:
        raise ValueError(f"decoder input {dec_dims[0]} != "
                         f"{MAX_NUM_TOKENS * TOKEN_DIM} + {PROP_DIM}")
    if dec_dims[-1] != NUM_ACTIONS:
        raise ValueError(f"decoder output {dec_dims[-1]} != {NUM_ACTIONS} actions")

    wrapper = FusedAsimovWrapper(encoder, decoder).eval()

    torch.manual_seed(args.seed)
    # Structured random inputs: tokenizer terms are O(1) (jpos rad, rad/s /
    # 6D-rot entries), proprioception likewise. 0.5 std covers the realistic
    # envelope while exercising FSQ codes away from a single bin.
    example = 0.5 * torch.randn(args.val_batch, ACTOR_OBS_DIM, dtype=torch.float32)

    with torch.no_grad():
        ref = wrapper(example).numpy()
    print(f"[export_asimov_onnx] PT wrapper action[0,:6] = {ref[0, :6]}", flush=True)

    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tmp_path = output_path + ".tmp"

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            example[:1],
            tmp_path,
            input_names=["obs"],
            output_names=["action"],
            dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
            opset_version=args.opset,
            do_constant_folding=False,  # don't fold-away FSQ rounding
            verbose=False,
        )
    print(f"[export_asimov_onnx] Wrote temporary ONNX to {tmp_path}", flush=True)

    import onnxruntime as ort
    sess = ort.InferenceSession(tmp_path, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    onnx_out = sess.run([out_name], {in_name: example.numpy()})[0]

    diff = np.abs(onnx_out - ref)
    max_d, mean_d = float(diff.max()), float(diff.mean())
    print(f"[export_asimov_onnx] Validation (PT fused vs ONNX, "
          f"batch={args.val_batch}):", flush=True)
    print(f"    max|onnx - pt|  = {max_d:.3e} rad", flush=True)
    print(f"    mean|onnx - pt| = {mean_d:.3e} rad", flush=True)

    if max_d > args.max_action_diff and not args.force:
        print(f"[export_asimov_onnx] FAILED: max diff {max_d:.3e} > "
              f"{args.max_action_diff:.1e}. Leaving {tmp_path} for inspection.",
              flush=True)
        sys.exit(1)

    os.replace(tmp_path, output_path)
    print(f"[export_asimov_onnx] OK -- promoted to {output_path} "
          f"({Path(output_path).stat().st_size / 1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
