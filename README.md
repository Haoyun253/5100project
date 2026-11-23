# PRDC 与 critic LayerNorm

本项目实现离线强化学习算法 PRDC，并支持在双 critic 的第一层、第二层或两层隐藏层上使用 LayerNorm。所有结构共用同一训练入口。算法基于 Ran 等人的 ICML 2023 论文 *Policy Regularization with Dataset Constraint for Offline Reinforcement Learning*。[论文](https://arxiv.org/abs/2306.06569) · [上游代码版本](https://github.com/LAMDA-RL/PRDC/tree/9f384640d52ff288e57ced09fc617617b557b3d1)

## 配置环境（Windows PowerShell）

安装 64 位 Python 3.12 和 Git，然后在 PowerShell 中运行：

```powershell
git clone https://github.com/Haoyun253/5100project.git
cd 5100project
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

选用一种 PyTorch 构建。普通 CPU 环境使用：

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
```

有 NVIDIA GPU 和兼容 CUDA 12.8 的驱动时可使用：

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
```

安装其余依赖并检查 MuJoCo、PyTorch 和设备：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "import gymnasium as gym, h5py, scipy, torch; env = gym.make('Hopper-v4'); print('Hopper observation:', env.reset(seed=0)[0].shape, 'PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available()); env.close()"
```

Hopper 的观测维度应为 `(11,)`。CPU 安装时，CUDA 检查返回 `False`。训练默认使用 CPU；GPU 安装检查通过后，在训练命令中添加 `--device cuda`。精确最近邻搜索使用 SciPy KDTree。

## 准备数据并验证训练入口

将 D4RL v2 HDF5 文件保存在本地。以下命令以 Hopper Medium Replay 为例，执行两次梯度更新和一回合评估，可验证数据读取、critic 与 actor 更新及 MuJoCo 评估。

```powershell
curl.exe --fail --location --create-dirs --output data/hopper_medium_replay-v2.hdf5 https://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/hopper_medium_replay-v2.hdf5
.\.venv\Scripts\python.exe train.py --data data/hopper_medium_replay-v2.hdf5 --task hopper --dataset-type medium-replay --critic layernorm --steps 2 --eval-freq 2 --eval-episodes 1
```

终端应输出 `step=2`、回合回报和标准化分数。训练其他数据集时，使 `--data`、`--task` 和 `--dataset-type` 与文件对应。运动控制支持 Hopper、Walker2d、HalfCheetah 和 Ant。使用 `train.py --help` 查看训练步数、随机种子、α 和保存权重等参数。

| `--critic` | 隐藏层变换 |
|---|---|
| `baseline` | 原版 PRDC 的两层 Linear → ReLU |
| `layernorm` | 两层均使用 Linear → LayerNorm → ReLU |
| `ln-first` / `ln-second` | 仅在第一层 / 第二层加入 LayerNorm |
| `ln-no-affine` | 两层均做标准化，固定缩放和平移 |
| `affine-only` | 两层均学习逐特征缩放和平移 |

## 可选：AntMaze

AntMaze 评估器读取 D4RL 仓库中的布局和 MuJoCo 模型。只运行运动控制时可跳过这一步。需要 AntMaze 时，在本仓库根目录执行：

```powershell
git clone https://github.com/Farama-Foundation/D4RL.git vendor/D4RL
git -C vendor/D4RL checkout 89141a689b0353b0dac3da5cba60da4b1b16254d
```

从对应 v2 注册项的 `dataset_url` 获取 HDF5 文件。训练时指定 `--task antmaze`，并将 `--dataset-type` 设为 `umaze`、`umaze-diverse`、`medium-play`、`medium-diverse`、`large-play` 或 `large-diverse`。评估使用本地 MuJoCo 绑定。[D4RL v2 注册表](https://github.com/Farama-Foundation/D4RL/blob/89141a689b0353b0dac3da5cba60da4b1b16254d/d4rl/locomotion/__init__.py)
