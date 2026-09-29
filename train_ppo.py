import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from unity_env import UnityCarEnv

# โฟลเดอร์สำหรับเซฟ Model และ Logs
models_dir = "models/PPO"
log_dir = "logs"
os.makedirs(models_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# สร้าง Environment
env = UnityCarEnv()

# เซฟ Checkpoint ทุกๆ 5,000 steps
checkpoint_callback = CheckpointCallback(
    save_freq=5000,
    save_path=models_dir,
    name_prefix="donkey_unity_model"
)

# ใช้ PPO ร่วมกับ CnnPolicy (ประมวลผลจากภาพกล้องโดยตรง)
model = PPO(
    policy="CnnPolicy",
    env=env,
    learning_rate=3e-4,
    n_steps=512,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    verbose=1,
    tensorboard_log=log_dir
)

print("Starting RL Training on Unity Environment...")
TIMESTEPS = 50000
model.learn(total_timesteps=TIMESTEPS, callback=checkpoint_callback, reset_num_timesteps=False)

# เซฟโมเดลรอบสุดท้าย
model.save(f"{models_dir}/final_model")
print("Training Complete! Final model saved.")

env.close()