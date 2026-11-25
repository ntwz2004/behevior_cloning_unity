import os
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from unity_env import UnityCarEnv

PROJECT_DIR = os.path.abspath(os.path.join(SRC_DIR, ".."))
MODEL_PATH = os.path.join(PROJECT_DIR, "models", "rl", "ppo_obstacle_normal.zip")

env = DummyVecEnv([lambda: UnityCarEnv(port=9091)])
model = PPO.load(MODEL_PATH)
print("[Info] Model loaded. Starting inference...")
obs = env.reset()

try:
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        
except KeyboardInterrupt:
    print("\n[Stop] User stopped the driving script.")
finally:
    env.close()
