import os
import sys
import time
from datetime import datetime

import cv2
import keyboard
import torch
import torch.nn as nn

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import dagger_log
from collect_bc import (
    DISCRETE_STEERING,
    OBS_RGB,
    THROTTLE,
    prepare_observation,
    to_bgr,
)
from model import NvidiaCNN
from unity_manual_env import UnityManualCarEnv

PROJECT_DIR = os.path.abspath(os.path.join(SRC_DIR, ".."))

# โฟลเดอร์เก็บภาพ DAgger ของ Unity (แบนราบ label_timestamp.jpg เหมือนฝั่ง Pi)
DAGGER_DIR = os.path.join(PROJECT_DIR, "datasets", "dagger_unity_3")
MODEL_FILE = os.path.join(PROJECT_DIR, "models", "bc", "model_unity_da_2.pth")

# โฟลเดอร์เก็บไฟล์ Excel: การรันแต่ละครั้งสร้างไฟล์ใหม่แยกกัน (เหมือนฝั่ง Pi)
LOG_DIR = os.path.join(PROJECT_DIR, "logs", "dagger_unity")

# ชื่อเรียก session นี้ (ใส่ในชื่อไฟล์) เช่น "round1" เว้นว่างได้
SESSION_NAME = "r3"

# 1 step ใน Unity = 0.1 s (fixedStepsPerAction=5 x 0.02 s) เท่ากับ car.CONTROL_PERIOD ฝั่ง Pi
# เวลาทั้งหมดใน log (เวลารวม, เวลากดแทรก) นับเป็น "เวลาใน Unity" = จำนวน step x 0.1
# จึงเทียบกับ log ของ Pi ได้ตรงๆ และไม่เพี้ยนแม้ Unity วิ่งช้า/เร็วกว่าเวลาจริง
CONTROL_PERIOD = 0.1

# ตอนผู้ใช้แทรก: เซฟภาพใหม่เมื่อ (1) เริ่มกดใหม่ หรือ (2) กดค้างและผ่านไปอย่างน้อยกี่วินาที (เวลา Unity)
# 0.3 = แตะ 1 ครั้งได้ 1 ภาพ, กดค้าง 1 วินาทีได้ ~4 ภาพ (เหมือนฝั่ง Pi)
SAVE_MIN_INTERVAL = 0.3

# output ของโมเดล: 0 = forward, 1 = left, 2 = right
AI_ACTIONS = {0: "forward", 1: "left", 2: "right"}

KEY_LABELS = {
    "up": "forward",
    "left": "left",
    "right": "right",
}

# แตะปุ่มสั้นกว่า 1 step ก็ไม่หลุด: จำปุ่มที่ถูกกดไว้จนกว่า step ถัดไปจะใช้ (เหมือน TAP_HOLD ฝั่ง Pi)
latched = {"label": None}

# แสดงภาพสดไม่ถี่เกินนี้ (วินาทีเวลาจริง)
UI_PERIOD = 0.1
DISPLAY_SCALE = 3


def on_key_down(event):
    label = KEY_LABELS.get(event.name)
    if label is not None:
        latched["label"] = label


def read_human_label():
    """ซ้าย/ขวามาก่อน forward (เหมือน collect_data) และรวมปุ่มที่แตะระหว่าง step"""
    label = None
    if keyboard.is_pressed("left"):
        label = "left"
    elif keyboard.is_pressed("right"):
        label = "right"
    elif keyboard.is_pressed("up"):
        label = "forward"
    elif latched["label"] is not None:
        label = latched["label"]

    latched["label"] = None
    return label


def draw_view(observation, mode, saved_count, dataset_total, override_seconds, ai_label, confidence):
    image = cv2.resize(
        to_bgr(observation),
        None,
        fx=DISPLAY_SCALE,
        fy=DISPLAY_SCALE,
        interpolation=cv2.INTER_NEAREST,
    )

    colors = {"BRAKE": (50, 50, 200), "OVERRIDE": (70, 200, 0), "AI DRIVING": (120, 120, 120)}
    color = colors[mode]
    image = cv2.copyMakeBorder(image, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=color)

    lines = [
        mode,
        f"AI: {ai_label or '-'} ({confidence * 100:4.1f}%)",
        f"Saved (session): {saved_count}",
        f"Dataset total: {dataset_total}",
        f"Override time: {override_seconds:6.1f} s",
    ]
    y = 28
    for line in lines:
        cv2.putText(image, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        y += 22

    cv2.imshow("DAgger Control Center (Unity)", image)
    cv2.waitKey(1)


def main():
    device = torch.device("cpu")
    model = NvidiaCNN().to(device)
    model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
    model.eval()
    softmax = nn.Softmax(dim=1)

    os.makedirs(DAGGER_DIR, exist_ok=True)
    env = UnityManualCarEnv()
    obs, _ = env.reset()

    keyboard.on_press(on_key_down)

    tracker = dagger_log.OverrideTracker()
    dataset_at_start = sum(dagger_log.count_dataset_images(DAGGER_DIR).values())

    saved_count = 0
    agree_count = 0
    last_saved_label = None
    last_save_time = 0.0
    steps = 0                   # จำนวน step ที่ผ่านไป: เวลา Unity = steps x CONTROL_PERIOD

    ai_label = None
    confidence = 0.0
    last_ui = 0.0

    session_start_dt = datetime.now()

    try:
        with torch.no_grad():
            while True:
                sim_time = steps * CONTROL_PERIOD

                if keyboard.is_pressed("q"):
                    print("\nExiting program via Q key...")
                    break

                # กด r เพื่อรีเซ็ตรถด้วยตัวเอง (ปิดช่วงกดแทรกที่ค้างอยู่)
                if keyboard.is_pressed("r"):
                    tracker.update(None, sim_time)
                    obs, _ = env.reset()
                    last_saved_label = None
                    latched["label"] = None
                    time.sleep(0.2)
                    continue

                observation = prepare_observation(obs)

                # โมเดลทำนายจากภาพที่ผู้ขับเห็นอยู่ตอนนี้ (ทุก step = 10 Hz เหมือนฝั่ง Pi)
                input_tensor = (
                    torch.from_numpy(observation.transpose((2, 0, 1)).copy())
                    .float()
                    .div(255.0)
                    .unsqueeze(0)
                    .to(device)
                )
                probabilities = softmax(model(input_tensor))[0]
                ai_choice = torch.argmax(probabilities).item()
                confidence = probabilities[ai_choice].item()
                ai_label = AI_ACTIONS[ai_choice]

                brake = keyboard.is_pressed("down")
                human_label = None if brake else read_human_label()

                tracker.update(human_label, sim_time)

                # เซฟภาพที่มนุษย์แก้ (ภาพก่อนสั่ง คู่กับ label ที่กด)
                if human_label is not None:
                    should_save = (
                        human_label != last_saved_label
                        or sim_time - last_save_time >= SAVE_MIN_INTERVAL
                    )

                    if should_save:
                        timestamp = int(time.time() * 1000)
                        path = os.path.join(DAGGER_DIR, f"{human_label}_{timestamp}.jpg")

                        if cv2.imwrite(path, to_bgr(observation)):
                            agreed = human_label == ai_label
                            saved_count += 1
                            if agreed:
                                agree_count += 1
                            tracker.add_image(agreed)

                            last_saved_label = human_label
                            last_save_time = sim_time
                else:
                    last_saved_label = None

                # ตัดสินใจคำสั่ง: เบรก > มนุษย์แทรก > โมเดล
                if brake:
                    steering, throttle = 0.0, 0.0
                    mode = "BRAKE"
                elif human_label is not None:
                    steering, throttle = DISCRETE_STEERING[human_label], THROTTLE
                    mode = "OVERRIDE"
                else:
                    steering, throttle = DISCRETE_STEERING[ai_label], THROTTLE
                    mode = "AI DRIVING"

                obs, _, _, _, _ = env.step([steering, throttle])
                steps += 1

                now = time.monotonic()
                if now - last_ui >= UI_PERIOD:
                    sim_time = steps * CONTROL_PERIOD
                    override_seconds = tracker.total_seconds(sim_time)
                    draw_view(
                        prepare_observation(obs), mode, saved_count,
                        dataset_at_start + saved_count, override_seconds,
                        ai_label, confidence,
                    )
                    if mode == "OVERRIDE":
                        status = (
                            f"[DAgger Override] {human_label:<7} | AI said {ai_label:<7}"
                            f" | saved={saved_count} | override={override_seconds:5.1f}s   "
                        )
                    elif mode == "BRAKE":
                        status = "[DAgger Override] Stopped                            "
                    else:
                        status = f"[AI Mode] {ai_label:<7} (Conf: {confidence * 100:5.1f}%)   "
                    print("\r" + status, end="", flush=True)
                    last_ui = now

    except KeyboardInterrupt:
        print("\nStopping manually...")

    finally:
        keyboard.unhook_all()
        total_seconds = steps * CONTROL_PERIOD
        session_end_dt = datetime.now()
        tracker.close(total_seconds)       # ปิดช่วงที่ค้างด้วยเวลา Unity (ไม่ปนกับนาฬิกาจริง)
        env.close()
        cv2.destroyAllWindows()

        override_seconds = sum(segment["duration"] for segment in tracker.segments)
        override_pct = override_seconds / total_seconds * 100.0 if total_seconds > 0 else 0.0
        dataset_total = sum(dagger_log.count_dataset_images(DAGGER_DIR).values())

        print(
            f"\n[DAgger] Session {total_seconds:.0f}s (Unity time) | "
            f"Override {override_seconds:.1f}s ({override_pct:.1f}%) in "
            f"{len(tracker.segments)} presses | "
            f"Saved {saved_count} | Dataset total {dataset_total}"
        )
        if saved_count > 0:
            print(f"[DAgger] AI agreed with human: {agree_count / saved_count * 100:.1f}%")

        try:
            path = dagger_log.save_excel(
                LOG_DIR,
                SESSION_NAME,
                DAGGER_DIR,
                session_start_dt,
                session_end_dt,
                total_seconds,
                tracker,
                saved_count,
                agree_count,
                None,
                {
                    "environment": "Unity",
                    "model": MODEL_FILE,
                    "throttle": THROTTLE,
                    "control_period": CONTROL_PERIOD,
                    "save_min_interval": SAVE_MIN_INTERVAL,
                },
                dataset_at_start,
            )
            if path:
                print(f"[Excel] Saved: {os.path.abspath(path)}")
        except Exception as exc:
            print(f"[Excel] Could not save log: {exc}")

        print("Closed all resources successfully.")


if __name__ == "__main__":
    main()
