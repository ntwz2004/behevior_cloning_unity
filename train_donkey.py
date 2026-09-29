import os
import gym
import gym_donkeycar
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

# ==========================================
# 1. ตั้งค่าการเชื่อมต่อกับ Donkey Simulator
# ==========================================
# รายชื่อสนาม: "donkey-generated-track-v0", "donkey-warehouse-v0", "donkey-circuit-launch-v0"
TRACK_NAME = "donkey-generated-track-v0"

# กำหนด Config ให้ส่งภาพขนาด 200x240 ตามมาตรฐานที่เราใช้
conf = {
    "exe_path": "remote",      # ใช้ "remote" เมื่อเปิดตัว Simulator เองล่วงหน้า
    "host": "127.0.0.1",
    "port": 9091,
    "body_style": "donkey",
    "body_rgb": (255, 0, 0),
    "car_name": "RL_Agent",
    "font_size": 10,
    "racer_name": "PPO_Bot",
    "country": "Thailand",
    "bio": "RL Benchmark Agent",
    "cam_resolution": (240, 200, 3), # (Width, Height, Channels)
    "img_w": 240,
    "img_h": 200,
    "img_d": 3,
}

# สร้าง Environment
env = gym.make(TRACK_NAME, conf=conf)

# ==========================================
# 2. ตั้งค่า Model PPO (CnnPolicy)
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on: {device}")

checkpoint_callback = CheckpointCallback(
    save_freq=10000, 
    save_path="./donkey_checkpoints/", 
    name_prefix="ppo_donkey"
)

# DonkeyCar ปกติมี Action 2 ค่า: [Steering (-1.0 to 1.0), Throttle (0.0 to 1.0)]
model = PPO(
    policy="CnnPolicy",
    env=env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    clip_range=0.2,
    verbose=1,
    device=device,
    tensorboard_log="./tensorboard_donkey/"
)

# ==========================================
# 3. เริ่มต้นการเรียนรู้ (Training Loop)
# ==========================================
print("\n--- Starting PPO Training on Donkey Simulator ---")
try:
    # เทรน 100,000 timesteps ขึ้นไป
    model.learn(total_timesteps=150000, callback=checkpoint_callback)
    model.save("ppo_donkey_final.zip")
    print("\nModel saved successfully as 'ppo_donkey_final.zip'!")
except KeyboardInterrupt:
    print("\nTraining interrupted! Saving current weights...")
    model.save("ppo_donkey_interrupted.zip")
finally:
    env.close()