import os
import sys
import time

import cv2
import keyboard

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from unity_manual_env import UnityManualCarEnv

# โฟลเดอร์แบนราบเหมือนฝั่ง Pi (dataset_bc): ชื่อไฟล์คือ label  เช่น left_1786428593300.jpg
# ใส่ชื่อ train_bc.py ได้เลย (อ่าน label จากชื่อไฟล์) ไม่ต้องใช้ data.csv
PROJECT_DIR = os.path.abspath(os.path.join(SRC_DIR, ".."))
DATASET_DIR = os.path.join(PROJECT_DIR, "datasets", "forward_obstacle")

#**
# forward_normal/ (เก็บภาพสนามปกติ ทิศไป)
# reverse_normal/ (เก็บภาพสนามปกติ ทิศกลับ)
# forward_obstacle/ (เก็บภาพสนามมีกล่อง ทิศไป)
# reverse_obstacle/ (เก็บภาพสนามมีกล่อง ทิศกลับ)

# ขนาดภาพ (กว้าง 120 x สูง 100) ต้องตรงกับ collect_data.py (Pi), drive_dagger.py และ RL
OBS_WIDTH = 120
OBS_HEIGHT = 100

# ลำดับสีที่เก็บลงไฟล์ ต้องตรงกับ OBS_RGB ฝั่ง Pi
#   False = BGR (cv2.imwrite จะได้สีถูกต้อง)   True = RGB
# unity_manual_env ส่งภาพ RGB มา
ENV_IS_RGB = True
OBS_RGB = False

# ค่าเดียวกับ car_config.DISCRETE_STEERING ฝั่ง Pi: ตรง = 0, ซ้าย = -1, ขวา = +1
DISCRETE_STEERING = {
    "forward": 0.0,
    "left": -1.0,
    "right": 1.0,
}

# ความเร็วคงที่ตอนมีปุ่มกด (เหมือน Pi ที่ความเร็วเฉลี่ยคงที่) ตรงกับ THROTTLE ใน unity_env.py
# ไม่มีถอยหลัง ปล่อยปุ่ม = หยุด
THROTTLE = 0.4

# เก็บภาพทุก step ที่มีปุ่มกด: 1 step = 0.1 s ใน Unity (10 Hz) เท่ากับ SAVE_PERIOD ฝั่ง Pi
# แสดงภาพสดไม่ถี่เกินนี้
UI_PERIOD = 0.1


def prepare_observation(frame):
    """ครอปกลางให้ได้สัดส่วนเดียวกับภาพเป้าหมาย แล้วย่อด้วย INTER_AREA (ตรรกะเดียวกับ car_config.prepare_observation)
    คืนภาพขนาด (OBS_HEIGHT, OBS_WIDTH) ลำดับสีตาม OBS_RGB"""
    height, width = frame.shape[:2]

    if (height, width) != (OBS_HEIGHT, OBS_WIDTH):
        target_ratio = OBS_WIDTH / OBS_HEIGHT

        if width / height > target_ratio:
            new_width = int(round(height * target_ratio))
            left = (width - new_width) // 2
            frame = frame[:, left:left + new_width]
        elif width / height < target_ratio:
            new_height = int(round(width / target_ratio))
            top = (height - new_height) // 2
            frame = frame[top:top + new_height, :]

        frame = cv2.resize(frame, (OBS_WIDTH, OBS_HEIGHT), interpolation=cv2.INTER_AREA)

    if ENV_IS_RGB != OBS_RGB:
        frame = frame[:, :, ::-1]

    return frame


def to_bgr(observation):
    """ภาพสำหรับ cv2.imwrite / cv2.imshow (ต้องเป็น BGR)"""
    return cv2.cvtColor(observation, cv2.COLOR_RGB2BGR) if OBS_RGB else observation


def main():
    os.makedirs(DATASET_DIR, exist_ok=True)
    env = UnityManualCarEnv()
    obs, _ = env.reset()

    saved_count = {"forward": 0, "left": 0, "right": 0}
    last_ui = 0.0

    try:
        while True:
            if keyboard.is_pressed("q"):
                break

            # กด r เพื่อรีเซ็ตด้วยตัวเอง
            if keyboard.is_pressed("r"):
                obs, _ = env.reset()
                time.sleep(0.2)
                continue

            # ซ้าย/ขวามีความสำคัญกว่า forward (เหมือนฝั่ง Pi)
            label = None
            if keyboard.is_pressed("left"):
                label = "left"
            elif keyboard.is_pressed("right"):
                label = "right"
            elif keyboard.is_pressed("up"):
                label = "forward"

            steering = DISCRETE_STEERING[label] if label else 0.0
            throttle = THROTTLE if label else 0.0

            # เก็บภาพที่ "เห็นก่อนตัดสินใจ" คู่กับ label ของปุ่มที่กดตอนนั้น
            # (เหมือนฝั่ง Pi: ภาพต่อ 1 label ที่ผู้ขับเห็น ไม่ใช่ภาพหลังรถขยับไปแล้ว)
            if label is not None:
                observation = prepare_observation(obs)
                timestamp = int(time.time() * 1000)
                path = os.path.join(DATASET_DIR, f"{label}_{timestamp}.jpg")

                if cv2.imwrite(path, to_bgr(observation)):
                    saved_count[label] += 1

            obs, _, _, _, _ = env.step([steering, throttle])

            now = time.monotonic()
            if now - last_ui >= UI_PERIOD:
                cv2.imshow("Collecting Data", to_bgr(prepare_observation(obs)))
                cv2.waitKey(1)
                print(
                    f"\r{label or 'stopped':<8} saved={saved_count}   ",
                    end="",
                    flush=True,
                )
                last_ui = now

    except KeyboardInterrupt:
        print("\nStopped manually.")

    finally:
        env.close()
        cv2.destroyAllWindows()
        print(f"\nSaved images: {saved_count}")
        print(f"Saved: {DATASET_DIR}")


if __name__ == "__main__":
    main()
