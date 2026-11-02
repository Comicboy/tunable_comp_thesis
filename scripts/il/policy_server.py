"""Serves a trained LeRobot policy over a local TCP socket, for rollouts in Isaac Lab.

Runs in the "lerobot" conda env (LeRobot 0.6.1, Python 3.12). The rollout script
(scripts/il/rollout_policy.py) runs in the Isaac Lab env and asks this server for one action per env
step. Two processes because the two envs cannot share one Python (see
claude/notes_imitation_learning.md, "Training setup").

    python scripts/il/policy_server.py --policy logs/il_train/act_v1/checkpoints/last/pretrained_model
    python scripts/il/policy_server.py --policy ... --n_action_steps 20        # re-plan every 20 steps
    python scripts/il/policy_server.py --policy ... --temporal_ensemble 0.01   # ACT paper's ensembling

The policy is run exactly as LeRobot runs it (as in lerobot-eval / lerobot-rollout): weights and
config with the policy class's from_pretrained, the pre-processor (adds the batch dimension, moves to
the GPU, normalises with the dataset statistics saved in the checkpoint) and the post-processor
(un-normalises the action, back to the CPU) loaded from the same checkpoint, then
policy.select_action. For ACT, select_action keeps a queue: it predicts a chunk of chunk_size actions,
returns the first n_action_steps of them one per call, and predicts again when the queue is empty.
With temporal ensembling it predicts a chunk at every call and returns a weighted average of all
chunks' predictions for this step (weights exp(-coeff * i), i = 0 for the oldest prediction).
n_action_steps and the ensembling only change inference, so they can be chosen here without retraining.

Protocol: one JSON object per line, request -> reply.
    {"cmd": "info"}                                       -> policy type, sizes, inference settings
    {"cmd": "reset"}                                      -> clears the action queue (call at every episode start)
    {"cmd": "act", "state": [48], "env_state": [3]}       -> {"action": [26], "infer_ms": ..., "replanned": ...}
One client at a time; when it disconnects the server waits for the next one (Ctrl+C to stop).
"""

import argparse
import json
import socket
import time
from pathlib import Path

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.policies.utils import prepare_observation_for_inference

parser = argparse.ArgumentParser(description="Serve a LeRobot policy to the Isaac Lab rollout script.")
parser.add_argument("--policy", required=True, help="checkpoint folder with config.json + model.safetensors "
                    "(…/checkpoints/<step>/pretrained_model)")
parser.add_argument("--port", type=int, default=9880, help="TCP port on 127.0.0.1")
parser.add_argument("--device", default="cuda")
parser.add_argument("--n_action_steps", type=int, default=None,
                    help="actions executed per predicted chunk (default: the checkpoint's, ACT: 100)")
parser.add_argument("--temporal_ensemble", type=float, default=None,
                    help="ACT temporal ensembling coefficient (e.g. 0.01); predicts every step (n_action_steps = 1)")
parser.add_argument("--task", default="Lift the cube 15 cm above the table and hold it there",
                    help="task string (as in the dataset; ACT does not use it)")
args = parser.parse_args()

STATE, ENV_STATE, ACTION = "observation.state", "observation.environment_state", "action"


def load_policy():
    path = str(Path(args.policy))
    cfg = PreTrainedConfig.from_pretrained(path)
    cfg.pretrained_path = path
    cfg.device = args.device
    # inference settings (do not change the trained weights)
    if args.temporal_ensemble is not None:
        cfg.temporal_ensemble_coeff = args.temporal_ensemble
        cfg.n_action_steps = 1
    elif args.n_action_steps is not None:
        if not 1 <= args.n_action_steps <= cfg.chunk_size:
            raise SystemExit(f"--n_action_steps must be in [1, {cfg.chunk_size}] (the chunk size)")
        cfg.n_action_steps = args.n_action_steps
    policy = get_policy_class(cfg.type).from_pretrained(path, config=cfg)
    policy.eval()
    pre, post = make_pre_post_processors(
        policy_cfg=cfg, pretrained_path=path,
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    return cfg, policy, pre, post


def main():
    cfg, policy, pre, post = load_policy()
    dims = {k: int(cfg.input_features[k].shape[0]) for k in (STATE, ENV_STATE)}
    dims[ACTION] = int(cfg.output_features[ACTION].shape[0])
    info = {
        "policy": cfg.type,
        "checkpoint": str(Path(args.policy).resolve()),
        "state_dim": dims[STATE],
        "env_state_dim": dims[ENV_STATE],
        "action_dim": dims[ACTION],
        "chunk_size": getattr(cfg, "chunk_size", None),
        "n_action_steps": getattr(cfg, "n_action_steps", None),
        "temporal_ensemble_coeff": getattr(cfg, "temporal_ensemble_coeff", None),
        "device": args.device,
    }
    print(f"[server] {json.dumps(info)}")
    device = torch.device(args.device)

    def reset():
        policy.reset()
        pre.reset()
        post.reset()

    def act(state, env_state):
        obs = {STATE: np.asarray(state, dtype=np.float32), ENV_STATE: np.asarray(env_state, dtype=np.float32)}
        if obs[STATE].shape != (dims[STATE],) or obs[ENV_STATE].shape != (dims[ENV_STATE],):
            raise ValueError(f"expected state {dims[STATE]} and env_state {dims[ENV_STATE]} numbers, "
                             f"got {obs[STATE].shape} and {obs[ENV_STATE].shape}")
        queue = getattr(policy, "_action_queue", None)
        replanned = queue is None or len(queue) == 0  # ACT predicts a new chunk on this call
        t0 = time.perf_counter()
        with torch.inference_mode():
            batch = prepare_observation_for_inference(obs, device, args.task, "ur5_wuji")
            batch = pre(batch)
            action = policy.select_action(batch)
            action = post(action)
        a = action.squeeze(0).detach().cpu().numpy().astype(float)
        return {"action": a.tolist(), "infer_ms": 1000 * (time.perf_counter() - t0), "replanned": bool(replanned)}

    # warm-up: the first call on the GPU is slow (~100 ms: kernels are loaded/compiled), so do it now
    warm = act([0.0] * dims[STATE], [0.0] * dims[ENV_STATE])
    reset()
    print(f"[server] warm-up call {warm['infer_ms']:.1f} ms")

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", args.port))
    srv.listen(1)
    print(f"[server] listening on 127.0.0.1:{args.port} (Ctrl+C to stop)")
    try:
        while True:
            conn, _ = srv.accept()
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  # no batching delay for small messages
            print("[server] client connected")
            reset()
            with conn, conn.makefile("rw", encoding="utf-8") as f:
                for line in f:
                    try:
                        req = json.loads(line)
                        cmd = req.get("cmd")
                        if cmd == "act":
                            rep = act(req["state"], req["env_state"])
                        elif cmd == "reset":
                            reset()
                            rep = {"ok": True}
                        elif cmd == "info":
                            rep = info
                        else:
                            rep = {"error": f"unknown cmd {cmd!r}"}
                    except Exception as exc:  # report to the client instead of dying
                        rep = {"error": f"{type(exc).__name__}: {exc}"}
                    f.write(json.dumps(rep) + "\n")
                    f.flush()
            print("[server] client disconnected")
    except KeyboardInterrupt:
        print("\n[server] stopped")
    finally:
        srv.close()


if __name__ == "__main__":
    main()