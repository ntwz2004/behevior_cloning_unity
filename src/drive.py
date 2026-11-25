"""
ให้โมเดล (BC / DAgger / RL) ขับรถใน Unity เฉยๆ ไม่มี log ไม่มี Excel

ปุ่ม: R = รีเซ็ตรถ | Q = ออก
ถ้า env ส่งสัญญาณจบรอบ (เช่นชน) จะรีเซ็ตให้เอง
เวลาจับโดย Unity เอง (ความละเอียด 0.02 s) แสดงเป็นทศนิยม 2 ตำแหน่ง
  - ผ่านครบรอบ: พิมพ์เวลาของรอบนั้น (รถวิ่งต่อไปได้เลย)
  - ทุกครั้งที่รีเซ็ต: พิมพ์เวลาที่วิ่งตั้งแต่เริ่มจนถึงตอนรีเซ็ต

ตัวอย่าง:
  python src/drive.py --type bc --model models/bc/model_bc.pth
  python src/drive.py --type rl --model models/rl/ppo_car_lap_complete.zip
"""

import argparse
import os
import sys
import time

import cv2
import keyboard
import numpy as np

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from unity_manual_env import UnityManualCarEnv

# ----------------------------------------------------------------------
# ค่าตั้ง (ต้องตรงกับตอนเทรน)
# ----------------------------------------------------------------------
OBS_WIDTH = 120
OBS_HEIGHT = 100
PROJECT_DIR = os.path.abspath(os.path.join(SRC_DIR, ".."))

# ภาพจาก env เป็น RGB
# ลำดับสีที่ส่งเข้า NvidiaCNN (BC/DAgger): False = BGR (ตรงกับ train_bc.py), True = RGB
OBS_RGB = False

THROTTLE = 0.4                  # ความเร็วคงที่ เท่ากับ THROTTLE ใน unity_env.py
STEP_PERIOD = 0.1               # 1 step ใน Unity = 0.1 s

# output ของ NvidiaCNN: 0 = forward, 1 = left, 2 = right
CLASS_STEERING = {0: 0.0, 1: -1.0, 2: 1.0}

SHOW_WINDOW = True


def prepare_observation(frame):
    """ครอปกลางให้ได้สัดส่วน 120:100 แล้วย่อด้วย INTER_AREA (ถ้าขนาดตรงอยู่แล้วไม่ทำอะไร)"""
    height, width = frame.shape[:2]

    if (height, width) != (OBS_HEIGHT, OBS_WIDTH):
        ratio = OBS_WIDTH / OBS_HEIGHT

        if width / height > ratio:
            new_width = int(round(height * ratio))
            left = (width - new_width) // 2
            frame = frame[:, left:left + new_width]
        elif width / height < ratio:
            new_height = int(round(width / ratio))
            top = (height - new_height) // 2
            frame = frame[top:top + new_height, :]

        frame = cv2.resize(frame, (OBS_WIDTH, OBS_HEIGHT), interpolation=cv2.INTER_AREA)

    return np.ascontiguousarray(frame)


class DiscretePolicy:
    """BC และ DAgger: NvidiaCNN เลือกคลาสที่มั่นใจสุด"""

    def __init__(self, model_file):
        import torch

        from model import NvidiaCNN

        self.torch = torch
        self.model = NvidiaCNN()
        self.model.load_state_dict(torch.load(model_file, map_location="cpu"))
        self.model.eval()

    def reset(self):
        pass

    def act(self, observation):
        image = observation if OBS_RGB else observation[:, :, ::-1]

        with self.torch.no_grad():
            tensor = (
                self.torch.from_numpy(image.transpose((2, 0, 1)).copy())
                .float()
                .div(255.0)
                .unsqueeze(0)
            )
            choice = int(self.torch.argmax(self.model(tensor), dim=1).item())

        return CLASS_STEERING[choice]


class RLPolicy:
    """PPO (ใช้ภาพเฟรมเดียว)"""

    def __init__(self, model_file):
        from stable_baselines3 import PPO

        try:
            self.model = PPO.load(model_file, device="cpu")
        except Exception as exc:
            print(f"[Model] Normal load failed: {exc}\n[Model] Retrying with custom_objects...")
            self.model = PPO.load(
                model_file,
                device="cpu",
                custom_objects={
                    "learning_rate": 0.0,
                    "lr_schedule": lambda _: 0.0,
                    "clip_range": lambda _: 0.0,
                },
            )

    def reset(self):
        pass

    def act(self, observation):
        action, _ = self.model.predict(observation, deterministic=True)
        return float(np.clip(np.asarray(action).reshape(-1)[0], -1.0, 1.0))


def main():
    parser = argparse.ArgumentParser(description="Let a trained model drive the car in Unity")
    parser.add_argument("--type", required=True, choices=["bc", "dagger", "rl"])
    parser.add_argument("--model", required=True, help=".pth สำหรับ BC/DAgger, .zip สำหรับ RL")
    args = parser.parse_args()

    if not os.path.isabs(args.model) and not os.path.exists(args.model):
        args.model = os.path.abspath(os.path.join(PROJECT_DIR, args.model))

    policy = RLPolicy(args.model) if args.type == "rl" else DiscretePolicy(args.model)

    env = UnityManualCarEnv()
    obs, _ = env.reset()
    policy.reset()

    flags = {"r": False, "q": False}

    def on_key_down(event):
        if event.name in flags:
            flags[event.name] = True      # จำปุ่มที่แตะไว้ แม้แตะสั้นกว่า 1 step

    keyboard.on_press(on_key_down)

    steps = 0
    run = 1
    steering = 0.0
    run_time = 0.0              # เวลาตั้งแต่ reset จาก Unity (วินาที)
    lap_times = []              # เวลาของรอบที่ครบในการวิ่งครั้งนี้ (หลัง reset แต่ละครั้งเริ่มใหม่)
    last_ui = 0.0
    run_start = time.monotonic()    # นาฬิกาจริงตั้งแต่เริ่มรอบ (ใช้แสดงผลให้ลื่นเท่านั้น)
    print(f"{args.type.upper()} driving | R = reset | Q = quit")

    try:
        while not flags["q"]:
            ended = None

            # ตรวจทั้งปุ่มที่แตะไว้ (hook) และปุ่มที่กดค้างอยู่ตอนนี้ เผื่อ hook ไม่ทำงานในบางเครื่อง
            if flags["r"] or keyboard.is_pressed("r"):
                ended = "reset (R)"
            else:
                observation = prepare_observation(obs)
                steering = policy.act(observation)
                obs, _, done, truncated, info = env.step([steering, THROTTLE])
                steps += 1

                # เวลาจาก Unity (ถ้า env ไม่ส่งมา ใช้ step x 0.1 แทน)
                run_time = info.get("run_time", steps * STEP_PERIOD)

                if info.get("event") == "lap_complete":
                    lap_times.append(info.get("lap_time", run_time))
                    print(f"\n  LAP {len(lap_times)}: {lap_times[-1]:.2f}s")

                if done or truncated:
                    ended = "ended by environment"

            if ended is not None:
                laps_text = f" | laps {len(lap_times)}" if lap_times else ""
                print(f"\nRun {run}: {ended} after {run_time:.2f}s ({steps} steps){laps_text}")
                obs, _ = env.reset()
                policy.reset()
                time.sleep(0.2)             # เว้นช่วงสั้นๆ ให้ Unity รีเซ็ตเสร็จก่อนรับปุ่มถัดไป (กด R ค้างจะรีเซ็ตซ้ำทุก 0.2 วินาที)
                flags["r"] = False
                run_start = time.monotonic()
                steps = 0
                run_time = 0.0
                lap_times = []
                run += 1
                continue

            now = time.monotonic()
            if now - last_ui >= 0.1:
                if SHOW_WINDOW:
                    shown = prepare_observation(obs)
                    cv2.imshow("Unity driving", shown[:, :, ::-1])      # env ส่ง RGB, imshow ต้องการ BGR
                    cv2.waitKey(1)
                # unity = เวลาจำลองจาก Unity (ขยับทีละ 0.10 s ตาม 1 step) ใช้เป็นผลวัด
                # real  = นาฬิกาจริง ขยับต่อเนื่อง ใช้ดูให้ลื่นเท่านั้น (ไม่เท่ากับเวลาใน Unity)
                print(
                    f"\r  run {run}  unity={run_time:7.2f}s  real={now - run_start:7.2f}s  "
                    f"steer={steering:+.2f}   ",
                    end="",
                    flush=True,
                )
                last_ui = now

    except KeyboardInterrupt:
        pass
    finally:
        keyboard.unhook_all()
        env.close()
        cv2.destroyAllWindows()
        print("\nClosed.")


if __name__ == "__main__":
    main()
