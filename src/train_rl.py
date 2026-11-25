import os
import sys
import torch

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import unity_env
from unity_env import UnityCarEnv
from train_log import TrainingRecorder
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


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
                        vec_env = self.model.get_vec_normalize_env()
                        if vec_env is not None:
                            vec_env.save(self.save_path + "_vecnormalize.pkl")
                        print(f"[Lap] Saved successful model: {self.save_path}.zip")
                    print("[Lap] Target reached. Stopping training.")
                    return False

            elif event in ("crash", "timeout"):
                if self.consecutive_laps > 0:
                    print(f"[Lap] {event}: consecutive-lap count reset.")
                self.consecutive_laps = 0

        return True


class EpisodeLogCallback(BaseCallback):
    """ส่งข้อมูลทุก episode (จำนวน step, reward, จบแบบไหน, checkpoint ที่ผ่าน) ให้ TrainingRecorder"""

    def __init__(self, recorder, verbose=0):
        super().__init__(verbose)
        self.recorder = recorder
        self.checkpoints = 0
        self.deviation_sum = 0.0
        self.deviation_steps = 0

    def _on_step(self):
        for info in self.locals.get("infos", []):
            event = info.get("event")

            if event == "checkpoint":
                self.checkpoints += 1
            elif event == "driving" and "deviation" in info:
                self.deviation_sum += float(info["deviation"])
                self.deviation_steps += 1

            episode = info.get("episode")  # Monitor ใส่ให้ตอนจบ episode: r = reward รวม, l = จำนวน step
            if episode is not None:
                mean_dev = self.deviation_sum / self.deviation_steps if self.deviation_steps else 0.0
                self.recorder.record_episode(
                    timestep=self.num_timesteps,
                    length=episode["l"],
                    reward=episode["r"],
                    event=event,
                    checkpoints=self.checkpoints,
                    mean_deviation=mean_dev,
                )
                self.checkpoints = 0
                self.deviation_sum = 0.0
                self.deviation_steps = 0

        return True


# 1. กำหนดโฟลเดอร์สำหรับเซฟ Model และ Tensorboard Logs
PROJECT_DIR = os.path.abspath(os.path.join(SRC_DIR, ".."))
models_dir = os.path.join(PROJECT_DIR, "models", "rl")
log_dir = os.path.join(PROJECT_DIR, "logs", "rl")
os.makedirs(models_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# ตรวจสอบอุปกรณ์ประมวลผล
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Hardware] Running on: {device.upper()}")
if device == "cpu":
    print("Recommend to use NVIDIA GPU for faster training")

# 2. เชื่อมต่อ Unity
#    - Monitor บันทึก Reward/Episode Length
#    - VecNormalize ปรับสเกล reward (ไม่ normalize ภาพ)

env = DummyVecEnv(
    [lambda: Monitor(UnityCarEnv(), filename=os.path.join(log_dir, "monitor.csv"))]
)
env = VecNormalize(env, norm_obs=False, norm_reward=True)

# 3. กำหนด Hyperparameters ของ PPO
model = PPO(
    policy="CnnPolicy",
    env=env,
    learning_rate=1e-4,
    n_steps=2048,
    batch_size=128,
    n_epochs=5,
    gamma=0.995,
    gae_lambda=0.95,
    ent_coef=0.005,
    target_kl=0.03,
    verbose=1,
    tensorboard_log=log_dir,
    device=device,
    seed=42,
)

# บันทึกสถิติการเทรน (ทุก episode ลง CSV ทันที และสรุปเป็น Excel ตอนจบ)
# RUN_NAME ใส่ในชื่อไฟล์ เช่น "run1" เว้นว่างได้
RUN_NAME = ""
recorder = TrainingRecorder(os.path.join(log_dir, "train_runs"), RUN_NAME)
episode_log_callback = EpisodeLogCallback(recorder)

# 4. บันทึก Checkpoint ทุกๆ 100,000 สเตป (พร้อมสถิติของ VecNormalize)
checkpoint_callback = CheckpointCallback(
    save_freq=100000,
    save_path=models_dir,
    name_prefix="ppo_car",
    save_vecnormalize=True,
)

# หยุดเมื่อวิ่งครบสนามติดกัน 3 รอบ (ไม่ใช่แค่สำเร็จครั้งเดียวแบบฟลุค)
lap_callback = StopOnConsistentLapCallback(
    required_consecutive_laps=3,
    save_path=os.path.join(models_dir, "ppo_car_lap_complete"),
)

# เพดานความปลอดภัย: ถ้ายังไม่ครบรอบภายในจำนวนนี้ ให้หยุดเพื่อตรวจ reward/สนาม
MAX_TRAINING_TIMESTEPS = 1000000

stop_reason = None   # ตั้งค่าตามสาเหตุที่จบ ถ้ายังเป็น None หลัง learn() แปลว่าจบเอง

print("RL Training...")
print("Ctrl+C to stop and save model")

try:
    model.learn(
        total_timesteps=MAX_TRAINING_TIMESTEPS,
        callback=[episode_log_callback, checkpoint_callback, lap_callback],
    )
except KeyboardInterrupt:
    stop_reason = "Interrupted by user (Ctrl+C)"
    print("\n[User Interrupted] Stopped...")
except ConnectionRefusedError:
    stop_reason = "Error: cannot connect to Unity"
    print("\n[Connection Error] Can't connect to Unity")
    print("Open Unity Editor and press Play button in the Scene before running the train command")
except TimeoutError:
    stop_reason = "Error: Unity did not respond (timeout)"
    print("\n[Timeout] Unity did not respond within 10 seconds")
    print("Check that Unity is still in Play mode and not paused or stuck")
except ConnectionError as e:
    stop_reason = f"Error: connection lost ({e})"
    print(f"\n[Connection Error] {e}")
except Exception as e:
    stop_reason = f"Error: {e}"
    print(f"\n [Error] {e}")
finally:
    model_paths = {}

    if hasattr(model, "num_timesteps") and model.num_timesteps > 0:
        final_path = os.path.join(models_dir, "ppo_car_final")
        model.save(final_path)
        env.save(os.path.join(models_dir, "ppo_car_final_vecnormalize.pkl"))
        model_paths["Final model"] = final_path + ".zip"
        print(f"[Done] Saved model ({model.num_timesteps} steps) at: {final_path}")

    lap_model = os.path.join(models_dir, "ppo_car_lap_complete.zip")
    if os.path.isfile(lap_model):
        model_paths["Lap-complete model"] = lap_model

    # สรุปสาเหตุที่หยุดเทรน
    if stop_reason is None:
        if lap_callback.consecutive_laps >= lap_callback.required_consecutive_laps:
            stop_reason = f"Target reached: {lap_callback.required_consecutive_laps} consecutive laps"
        else:
            stop_reason = f"Reached MAX_TRAINING_TIMESTEPS ({MAX_TRAINING_TIMESTEPS})"

    # ค่าตั้งที่เก็บลง Excel เพื่อให้ย้อนดูได้ว่าการเทรนรอบนี้ใช้ค่าอะไร
    settings = {
        "device": device,
        "seed": getattr(model, "seed", None),
        "learning_rate": getattr(model, "learning_rate", None),
        "n_steps": getattr(model, "n_steps", None),
        "batch_size": getattr(model, "batch_size", None),
        "n_epochs": getattr(model, "n_epochs", None),
        "gamma": getattr(model, "gamma", None),
        "gae_lambda": getattr(model, "gae_lambda", None),
        "ent_coef": getattr(model, "ent_coef", None),
        "target_kl": getattr(model, "target_kl", None),
        "MAX_TRAINING_TIMESTEPS": MAX_TRAINING_TIMESTEPS,
        "frame stack": 1,
        "required consecutive laps": lap_callback.required_consecutive_laps,
    }
    for name in (
        "OBS_WIDTH", "OBS_HEIGHT", "MAX_EPISODE_STEPS", "THROTTLE", "REWARD_DRIVING",
        "STEERING_PENALTY", "REWARD_CHECKPOINT", "REWARD_CRASH", "REWARD_LAP_COMPLETE",
    ):
        if hasattr(unity_env, name):
            settings[f"unity_env.{name}"] = getattr(unity_env, name)

    try:
        path = recorder.save_excel(
            getattr(model, "num_timesteps", 0), stop_reason, settings, model_paths
        )
        print(
            f"[Training] {getattr(model, 'num_timesteps', 0)} timesteps | "
            f"{len(recorder.rows)} episodes | {stop_reason}"
        )
        if path:
            print(f"[Excel] Saved: {os.path.abspath(path)}")
    except Exception as exc:
        recorder.close_csv()
        print(f"[Excel] Could not save log: {exc} (episodes are in {recorder.csv_path})")

    env.close()