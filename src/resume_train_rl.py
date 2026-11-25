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

            episode = info.get("episode")
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

# ระบุชื่อไฟล์ที่ต้องการโหลดมาเทรนต่อ (แก้ชื่อให้ตรงกับไฟล์ล่าสุดที่คุณมี)
PROJECT_DIR = os.path.abspath(os.path.join(SRC_DIR, ".."))
RL_MODELS_DIR = os.path.join(PROJECT_DIR, "models", "rl")
LOAD_MODEL_PATH = os.path.join(RL_MODELS_DIR, "ppo_1.zip")
LOAD_VECNORM_PATH = os.path.join(RL_MODELS_DIR, "ppo_car_lap_complete_vecnormalize.pkl")

models_dir = os.path.join(PROJECT_DIR, "models", "rl", "resumed")
log_dir = os.path.join(PROJECT_DIR, "logs", "rl", "resumed")
os.makedirs(models_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Hardware] Running on: {device.upper()}")

env = DummyVecEnv(
    [lambda: Monitor(UnityCarEnv(), filename=os.path.join(log_dir, "monitor_resume.csv"))]
)

# โหลดสถิติ VecNormalize เดิมกลับมา เพื่อให้โมเดลคุ้นเคยกับสเกล Reward เดิม
if os.path.exists(LOAD_VECNORM_PATH):
    env = VecNormalize.load(LOAD_VECNORM_PATH, env)
    env.training = True      # ต้องตั้งให้เป็น True เพื่อให้อัปเดตสถิติ Reward ต่อจากการเทรนเดิม
    env.norm_reward = True
    print(f"[Info] Loaded VecNormalize stats from: {LOAD_VECNORM_PATH}")
else:
    raise FileNotFoundError(f"[Error] VecNormalize .pkl file not found at {LOAD_VECNORM_PATH}!")

if os.path.exists(LOAD_MODEL_PATH):
    # โหลดโมเดลเดิม พร้อมกำหนด tensorboard_log ให้เขียนลง log_dir ของรอบที่เทรนต่อ
    model = PPO.load(LOAD_MODEL_PATH, env=env, device=device, tensorboard_log=log_dir)
    print(f"[Info] Successfully loaded model from: {LOAD_MODEL_PATH}")
else:
    raise FileNotFoundError(f"[Error] Model .zip file not found at {LOAD_MODEL_PATH}!")


RUN_NAME = "resumed"
recorder = TrainingRecorder(os.path.join(log_dir, "train_runs"), RUN_NAME)
episode_log_callback = EpisodeLogCallback(recorder)

checkpoint_callback = CheckpointCallback(
    save_freq=100000,
    save_path=models_dir,
    name_prefix="ppo_car_resumed",
    save_vecnormalize=True,
)

lap_callback = StopOnConsistentLapCallback(
    required_consecutive_laps=10,
    save_path=os.path.join(models_dir, "ppo_car_resumed_lap_complete"),
)

MAX_TRAINING_TIMESTEPS = 1000000
stop_reason = None

print("Resuming RL Training...")
print("Ctrl+C to stop and save model")

try:
    # เริ่มเทรนต่อ
    model.learn(
        total_timesteps=MAX_TRAINING_TIMESTEPS,
        callback=[episode_log_callback, checkpoint_callback, lap_callback],
        reset_num_timesteps=False, # **สำคัญมาก** ปิดการรีเซ็ต Step เพื่อให้ TensorBoard วิ่งต่อเนื่องจากเดิม
        tb_log_name="PPO"          # ให้เซฟกราฟลงโฟลเดอร์เดิม (PPO_1, PPO_2...)
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
        final_path = os.path.join(models_dir, "ppo_car_resumed_final")
        model.save(final_path)
        env.save(os.path.join(models_dir, "ppo_car_resumed_final_vecnormalize.pkl"))
        model_paths["Final model"] = final_path + ".zip"
        print(f"[Done] Saved resumed model at: {final_path}")

    lap_model = os.path.join(models_dir, "ppo_car_resumed_lap_complete.zip")
    if os.path.isfile(lap_model):
        model_paths["Lap-complete model"] = lap_model

    if stop_reason is None:
        if lap_callback.consecutive_laps >= lap_callback.required_consecutive_laps:
            stop_reason = f"Target reached: {lap_callback.required_consecutive_laps} consecutive laps"
        else:
            stop_reason = f"Reached MAX_TRAINING_TIMESTEPS ({MAX_TRAINING_TIMESTEPS})"

    settings = {
        "resumed_from": LOAD_MODEL_PATH,
        "vecnormalize_from": LOAD_VECNORM_PATH,
        "device": device,
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
