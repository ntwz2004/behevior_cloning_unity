import os
import torch
from unity_env import UnityCarEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor


class StopOnConsistentLapCallback(BaseCallback):
    """Stop after the training policy completes consecutive full laps."""

    def __init__(self, required_consecutive_laps=3, save_path=None, verbose=1):
        super().__init__(verbose)
        self.required_consecutive_laps = required_consecutive_laps
        self.save_path = save_path
        self.consecutive_laps = 0
        self.total_laps = 0

    def _on_step(self):
        for info in self.locals.get("infos", []):
            event = info.get("event")

            if event == "lap_complete":
                self.total_laps += 1
                self.consecutive_laps += 1
                print(
                    "[Lap] Complete "
                    f"({self.consecutive_laps}/"
                    f"{self.required_consecutive_laps} consecutive, "
                    f"{self.total_laps} total)"
                )

                if self.consecutive_laps >= self.required_consecutive_laps:
                    if self.save_path:
                        self.model.save(self.save_path)
                        print(f"[Lap] Saved successful model: {self.save_path}.zip")
                    print("[Lap] Target reached. Stopping training.")
                    return False

            elif event == "crash":
                if self.consecutive_laps > 0:
                    print("[Lap] Crash: consecutive-lap count reset.")
                self.consecutive_laps = 0

        return True

# 1. กำหนดโฟลเดอร์สำหรับเซฟ Model และ Tensorboard Logs
models_dir = os.path.abspath("Model/RL_Models")
log_dir = os.path.abspath("logs")
os.makedirs(models_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# ตรวจสอบอุปกรณ์ประมวลผล
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Hardware] Running on: {device.upper()}")
if device == "cpu":
    print("Recommend to use NVIDIA GPU for faster training")

# 2. เชื่อมต่อ Unity และครอบด้วย Monitor เพื่อบันทึก Reward/Episode Length ลง Tensorboard
raw_env = UnityCarEnv()
env = Monitor(raw_env, filename=os.path.join(log_dir, "monitor.csv"))

# 3. กำหนด Hyperparameters ของ PPO
model = PPO(
    policy="CnnPolicy",
    env=env,
    learning_rate=3e-5,
    n_steps=2048,
    batch_size=64,
    n_epochs=5,         
    gamma=0.99,
    ent_coef=0.01,
    target_kl=None,     
    verbose=1,
    tensorboard_log=log_dir,
    device=device,
    seed=42,
)

# 4. บันทึก Checkpoint ทุกๆ 10,000 สเตป
checkpoint_callback = CheckpointCallback(
    save_freq=10000, 
    save_path=models_dir, 
    name_prefix="ppo_car"
)

# หยุดเมื่อวิ่งครบสนามติดกัน 3 รอบ (ไม่ใช่แค่สำเร็จครั้งเดียวแบบฟลุค)
lap_callback = StopOnConsistentLapCallback(
    required_consecutive_laps=3,
    save_path=os.path.join(models_dir, "ppo_car_lap_complete"),
)

# เพดานความปลอดภัย: ถ้ายังไม่ครบรอบภายในจำนวนนี้ ให้หยุดเพื่อตรวจ reward/สนาม
MAX_TRAINING_TIMESTEPS = 1_000_000

print("RL Training...")
print("Ctrl+C to stop and save model")

try:
    model.learn(
        total_timesteps=MAX_TRAINING_TIMESTEPS,
        callback=[checkpoint_callback, lap_callback],
    )
except KeyboardInterrupt:
    print("\n[User Interrupted] Stopped...")
except ConnectionRefusedError:
    print("\n[Connection Error] Can't connect to Unity")
    print("Open Unity Editor and press Play button in the Scene before running the train command")
except Exception as e:
    print(f"\n [Error] {e}")
finally:
    if hasattr(model, "num_timesteps") and model.num_timesteps > 0:
        final_path = os.path.join(models_dir, "ppo_car_final")
        model.save(final_path)
        print(f"[Done] Saved model ({model.num_timesteps} steps) at: {final_path}")
    env.close()
