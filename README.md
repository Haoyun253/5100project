# PRDC and Critic LayerNorm

This repository contains two implementations based on *Policy Regularization with Dataset Constraint for Offline Reinforcement Learning* by Ran et al. (ICML 2023). [Paper](https://arxiv.org/abs/2306.06569) · [Upstream source](https://github.com/LAMDA-RL/PRDC/tree/9f384640d52ff288e57ced09fc617617b557b3d1)

| Directory | Contents |
|---|---|
| [PRDC/](PRDC/) | The authors' original implementation at commit `9f384640d52ff288e57ced09fc617617b557b3d1`. |
| [PRDC-LayerNorm/](PRDC-LayerNorm/) | A modified copy of the same commit. `net/critic.py` implements LayerNorm for the twin critics. `prdc.py` selects the critic configuration. By default, each critic uses LayerNorm in both hidden layers. |

Each directory includes its own `main.py` and the authors' dependency list for the legacy Gym/D4RL stack. For the Windows setup below, use `requirements.txt` and `train.py` in the repository root. This entry point handles data loading and evaluation for both implementations. Use `--critic baseline` to load `PRDC/` or `--critic layernorm` to load `PRDC-LayerNorm/`.

## Environment setup (Windows PowerShell)

Install 64-bit Python 3.12 and Git, then run these commands in PowerShell.

```powershell
git clone https://github.com/Haoyun253/5100project.git
cd 5100project
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

Choose the PyTorch build for your hardware. For CPU training, install the CPU build.

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
```

For an NVIDIA GPU with a CUDA 12.8-compatible driver, install the CUDA build instead.

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
```

Install the remaining dependencies, then check that MuJoCo can reset Hopper and that PyTorch detects your device.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "import gymnasium as gym, h5py, scipy, torch; env = gym.make('Hopper-v4'); print('Hopper observation:', env.reset(seed=0)[0].shape, 'PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available()); env.close()"
```

The Hopper observation shape should be `(11,)`. The CUDA check returns `False` for a CPU installation. Training defaults to CPU. If the check returns `True`, add `--device cuda` to a training command to use your GPU. Exact nearest-neighbor search uses SciPy's KDTree.

## Download a dataset and verify training

Download the D4RL v2 Hopper Medium Replay HDF5 file, then run both implementations on it. Each command runs two training steps and one evaluation episode to check data loading, critic and actor updates, and MuJoCo evaluation.

```powershell
curl.exe --fail --location --create-dirs --output data/hopper_medium_replay-v2.hdf5 https://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/hopper_medium_replay-v2.hdf5
.\.venv\Scripts\python.exe train.py --data data/hopper_medium_replay-v2.hdf5 --task hopper --dataset-type medium-replay --critic baseline --steps 2 --eval-freq 2 --eval-episodes 1
.\.venv\Scripts\python.exe train.py --data data/hopper_medium_replay-v2.hdf5 --task hopper --dataset-type medium-replay --critic layernorm --steps 2 --eval-freq 2 --eval-episodes 1
```

Each run should print `step=2`, the episode return, and the normalized score. For another dataset, set `--data`, `--task`, and `--dataset-type` to match the file. Supported locomotion tasks include Hopper, Walker2d, HalfCheetah, and Ant. Use `train.py --help` for options such as the training budget, random seed, alpha, and checkpoint path.

| `--critic` | Hidden-layer transformation |
|---|---|
| `baseline` | Original PRDC. Both hidden layers use Linear → ReLU. |
| `layernorm` | Both hidden layers use Linear → LayerNorm → ReLU. |
| `ln-first` / `ln-second` | Apply LayerNorm only in the first or second hidden layer, respectively. |
| `ln-no-affine` | Normalize both hidden layers with fixed scale and shift. |
| `affine-only` | Apply only a learned per-feature scale and shift in both hidden layers. |

## Optional AntMaze setup

The AntMaze evaluator reads the maze layouts and MuJoCo model from the D4RL repository. If you only run locomotion tasks, you can skip this step. To use AntMaze, run the following commands from the repository root.

```powershell
git clone https://github.com/Farama-Foundation/D4RL.git vendor/D4RL
git -C vendor/D4RL checkout 89141a689b0353b0dac3da5cba60da4b1b16254d
```

Download the HDF5 file from the `dataset_url` in the matching D4RL v2 registration. Run training with `--task antmaze` and set `--dataset-type` to `umaze`, `umaze-diverse`, `medium-play`, `medium-diverse`, `large-play`, or `large-diverse`. Evaluation uses the local MuJoCo bindings. [D4RL v2 registrations](https://github.com/Farama-Foundation/D4RL/blob/89141a689b0353b0dac3da5cba60da4b1b16254d/d4rl/locomotion/__init__.py)
