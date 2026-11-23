"""Windows MuJoCo evaluator for the six pinned D4RL AntMaze-v2 registrations."""
import ast
import xml.etree.ElementTree as ET
from functools import lru_cache

import gymnasium as gym
import mujoco
import numpy as np

from pathlib import Path

LOCOMOTION = Path(__file__).resolve().parent / "vendor/D4RL/d4rl/locomotion"


def maze_registrations():
    registrations = {}
    for node in ast.parse((LOCOMOTION / "__init__.py").read_text()).body:
        if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
            continue
        call = {arg.arg: arg.value for arg in node.value.keywords}
        name = ast.literal_eval(call["id"])
        if not name.endswith("-v2"):
            continue
        params = dict(zip([ast.literal_eval(k) for k in call["kwargs"].keys], call["kwargs"].values))
        registrations[name] = (params["maze_map"].attr, ast.literal_eval(call["max_episode_steps"]),
                               ast.literal_eval(params["dataset_url"]).replace("http:", "https:"))
    return registrations


@lru_cache
def maze_model(dataset):
    layout_name, horizon, _ = maze_registrations()[dataset]
    # Read the actual D4RL test layout without importing its legacy Gym runtime.
    assignments = ast.parse((LOCOMOTION / "maze_env.py").read_text()).body
    layout = next(node.value for node in assignments if isinstance(node, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == layout_name for t in node.targets))
    rows = [[cell.id if isinstance(cell, ast.Name) else ast.literal_eval(cell)
             for cell in row.elts] for row in layout.elts]
    xml = ET.parse(LOCOMOTION / "assets/ant.xml")
    world = xml.find(".//worldbody")
    start_y, start_x = next((i, j) for i, row in enumerate(rows) for j, x in enumerate(row) if x == "R")
    goals = []
    for i, row in enumerate(rows):
        for j, cell in enumerate(row):
            x, y = 4 * (j - start_x), 4 * (i - start_y)
            if cell == 1:
                ET.SubElement(world, "geom", name=f"block_{i}_{j}", pos=f"{x} {y} 1", size="2 2 1",
                              type="box", contype="1", conaffinity="1", rgba="0.7 0.5 0.3 1")
            elif cell == "G":
                goals.append((x, y))
    model = mujoco.MjModel.from_xml_string(ET.tostring(xml.getroot(), encoding="unicode"))
    return model, np.asarray(goals), horizon


class AntMazeEval(gym.Env):
    def __init__(self, dataset):
        self.model, self.goals, self.horizon = maze_model(dataset)
        self.data = mujoco.MjData(self.model)

    def observation(self):
        return np.concatenate((self.data.qpos, self.data.qvel))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        # MazeEnv.goal_sampler uses two successive positive jitters per coordinate.
        cell = self.goals[rng.integers(len(self.goals))]
        self.goal = np.maximum(cell + rng.uniform(0, 1, 2) + rng.uniform(0, 0.5, 2), 0)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] += rng.uniform(-0.1, 0.1, self.model.nq)
        self.data.qvel[:] = rng.normal(0, 0.1, self.model.nv)
        mujoco.mj_forward(self.model, self.data)
        self.steps = 0
        return self.observation(), {}

    def step(self, action):
        low, high = self.model.actuator_ctrlrange.T
        self.data.ctrl[:] = np.clip(low + (action + 1) * 0.5 * (high - low), low, high)
        mujoco.mj_step(self.model, self.data, nstep=5)
        self.steps += 1
        success = bool(np.linalg.norm(self.data.qpos[:2] - self.goal) <= 0.5)
        return self.observation(), float(success), success, self.steps >= self.horizon, {}
