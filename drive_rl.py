from pathlib import Path

import cv2
from stable_baselines3 import PPO

from unity_env import UnityCarEnv


# เลือก checkpoint ที่ต้องการทดสอบ (250k steps)
MODEL_PATH = (
    Path(__file__).resolve().parent
    / "Model"
    / "RL_Models"
    / "Best2.zip"
)   

if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"File not found: {MODEL_PATH}")

print(f"Loading model: {MODEL_PATH}")
model = PPO.load(str(MODEL_PATH))

env = UnityCarEnv()
obs, _ = env.reset()

print("Starting test drive")
print("Q for quit")

try:
    while True:
        # deterministic=True: ใช้ action ที่โมเดลมั่นใจที่สุด
        action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, _ = env.step(action)

        cv2.imshow(
            "AI Driving Camera",
            cv2.cvtColor(obs, cv2.COLOR_RGB2BGR),
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # รองรับทั้งชน และกรณีในอนาคตที่ตั้ง time limit กลับมา
        if terminated or truncated:
            reason = "CRASH / OFFTRACK" if terminated else "TIME LIMIT"
            print(f"Restarting: {reason}")
            obs, _ = env.reset()

finally:
    env.close()
    cv2.destroyAllWindows()
    print("Closed Unity connection and test drive")