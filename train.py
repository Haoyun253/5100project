"""Train PRDC on a local D4RL v2 HDF5 dataset."""
import argparse
import sys
from pathlib import Path
from typing import NamedTuple

import gymnasium as gym
import h5py
import numpy as np
import torch

MOTION_TYPES = ("random", "medium-replay", "medium-expert")
MAZE_TYPES = ("umaze", "umaze-diverse", "medium-play", "medium-diverse", "large-play", "large-diverse")
CRITICS = ("baseline", "layernorm", "ln-first", "ln-second", "ln-no-affine", "affine-only")


class Task(NamedTuple):
    env: str
    alpha: float
    ref_min: float
    ref_max: float


# D4RL v2 locomotion score anchors and upstream PRDC policy weights.
TASKS = {
    "hopper": Task("Hopper-v4", 2.5, -20.272305, 3234.3),
    "walker2d": Task("Walker2d-v4", 2.5, 1.629008, 4592.3),
    "halfcheetah": Task("HalfCheetah-v4", 40.0, -280.178953, 12135.0),
    # Ant is an extension: use the Walker2d alpha for both variants.
    "ant": Task("Ant-v4", 2.5, -325.6, 3879.7),
    "antmaze": Task("D4RL AntMaze-v2 (native MuJoCo)", 2.5, 0, 1),
}


def task_parameters(task, dataset_type):
    if task == "antmaze":
        if dataset_type not in MAZE_TYPES:
            raise ValueError("AntMaze requires a registered maze dataset")
        size = dataset_type.split("-")[0]
        alpha, beta = {"umaze": (2.5, 2.0), "medium": (7.5, 7.5), "large": (20.0, 15.0)}[size]
        return alpha, beta, 10000, -1
    if dataset_type not in (*MOTION_TYPES, "medium"):
        raise ValueError("Locomotion requires a locomotion dataset")
    return TASKS[task].alpha, 2.0, 1, 0

def load_transitions(path, task):
    # Match D4RL qlearning_dataset(env, terminate_on_end=False) for this v2 file.
    with h5py.File(path, "r") as data:
        if task == "ant":
            # All 84 contact channels in this D4RL file are zero. Ant-v4
            # exposes the same leading 27 position/velocity features by default.
            observations = data["observations"][:, :27].astype(np.float32)
        else:
            observations = data["observations"][:].astype(np.float32)
        keep = ~data["timeouts"][:-1].astype(bool)
        return {
            "observations": observations[:-1][keep],
            "next_observations": observations[1:][keep],
            "actions": data["actions"][:-1][keep].astype(np.float32),
            "rewards": data["rewards"][:-1][keep].astype(np.float32),
            "terminals": data["terminals"][:-1][keep].astype(bool),
        }


def evaluate(policy, mean, std, seed, episodes, task="hopper", dataset_type="medium"):
    rewards = []
    spec = TASKS[task]
    if task == "antmaze":
        from antmaze_eval import AntMazeEval
        evaluator = AntMazeEval(f"antmaze-{dataset_type}-v2")
    else:
        evaluator = gym.make(spec.env)
    with evaluator as env:
        for episode in range(episodes):
            obs, _ = env.reset(seed=seed + 100 if episode == 0 else None)
            total = 0.0
            while True:
                action = policy.select_action((obs - mean) / std)
                obs, reward, terminated, truncated, _ = env.step(action)
                total += reward
                if terminated or truncated:
                    break
            rewards.append(total)
    avg = float(np.mean(rewards))
    score = 100 * (avg - spec.ref_min) / (spec.ref_max - spec.ref_min)
    return avg, score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="Local D4RL v2 HDF5 file")
    parser.add_argument("--task", choices=TASKS, default="hopper")
    parser.add_argument("--dataset-type", default="medium")
    parser.add_argument("--critic", choices=CRITICS, default="baseline")
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--eval-freq", type=int, default=1000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--checkpoint", type=Path, help="Optional output file for model and optimizer weights")
    args = parser.parse_args()
    version = "PRDC" if args.critic == "baseline" else "PRDC-LayerNorm"
    sys.path.insert(0, str(Path(__file__).resolve().parent / version))
    from prdc import PRDC
    from utils.buffer import ReplayBuffer
    if min(args.steps, args.eval_freq, args.eval_episodes) < 1:
        parser.error("steps, eval-freq and eval-episodes must be positive")
    if not args.data.is_file():
        parser.error(f"Missing dataset: {args.data}")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable. Check the PyTorch build and driver, or use --device cpu.")
    if args.task == "antmaze" and not (Path(__file__).resolve().parent / "vendor/D4RL/d4rl/locomotion/assets/ant.xml").is_file():
        parser.error("AntMaze needs the D4RL assets. See README.md.")
    try:
        alpha, beta, scale, shift = task_parameters(args.task, args.dataset_type)
    except ValueError as exc:
        parser.error(str(exc))
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dataset = load_transitions(args.data, args.task)
    state_dim = dataset["observations"].shape[1]
    action_dim = dataset["actions"].shape[1]
    buffer = ReplayBuffer(state_dim, action_dim, args.device, f"{args.task}-{args.dataset_type}-v2",
                          scale, shift, max_size=0)
    buffer.convert_D4RL(dataset)
    del dataset
    mean, std = buffer.normalize_states()
    points = np.hstack([beta * buffer.state, buffer.action])
    critic_options = {} if version == "PRDC" else {"critic_mode": args.critic}
    policy = PRDC(points, state_dim, action_dim, 1.0, args.device, **critic_options,
                  alpha=alpha if args.alpha is None else args.alpha, beta=beta)
    for step in range(1, args.steps + 1):
        policy.train(buffer, 256)
        if step % args.eval_freq == 0 or step == args.steps:
            avg, score = evaluate(policy, mean, std, args.seed, args.eval_episodes, args.task, args.dataset_type)
            print(f"step={step} return={avg:.3f} score={score:.3f}", flush=True)
    if args.checkpoint:
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        policy.save(args.checkpoint)
        np.savez(str(args.checkpoint) + ".norm.npz", mean=mean, std=std)


if __name__ == "__main__":
    main()


