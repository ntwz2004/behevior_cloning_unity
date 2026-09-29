import os
import time
import csv
import cv2
import keyboard
from unity_env import UnityCarEnv

# ตั้งค่าโฟลเดอร์ให้ตรงกับโครงสร้างใหม่
base_dataset_dir = os.path.join("All Dataset", "dataset_image_unity")
img_dir = os.path.join(base_dataset_dir, "images")
os.makedirs(img_dir, exist_ok=True)
csv_path = os.path.join(base_dataset_dir, "data.csv")

env = UnityCarEnv()
obs, _ = env.reset()

print("========================================")
print("เริ่มเก็บข้อมูล: เดินหน้า (↑), ถอยหลัง (↓), เลี้ยว (←/→)")
print("[r] รีเซ็ตรถกลับจุดสตาร์ต")
print("[q] หยุดและบันทึกข้อมูล")
print("========================================")

file_exists = os.path.exists(csv_path)
csv_file = open(csv_path, mode='a', newline='')
writer = csv.writer(csv_file)
if not file_exists:
    writer.writerow(["image_name", "steering", "throttle"])

try:
    while True:
        steering = 0.0
        throttle = 0.0

        if keyboard.is_pressed('up'):
            throttle = 1.0
        elif keyboard.is_pressed('down'):
            throttle = -0.6

        if keyboard.is_pressed('left'):
            steering = -0.8
        elif keyboard.is_pressed('right'):
            steering = 0.8

        # กด r เพื่อรีเซ็ตด้วยตัวเอง (ไม่ตัดจบอัตโนมัติแล้ว)
        if keyboard.is_pressed('r'):
            obs, _ = env.reset()
            time.sleep(0.2)
            continue

        if keyboard.is_pressed('q'):
            break

        obs, reward, done, truncated, _ = env.step([steering, throttle])

        # บันทึกภาพเฉพาะเมื่อมีการขยับรถ (คันเร่ง หรือ พวงมาลัย ไม่เป็น 0)
        if throttle != 0 or steering != 0:
            
            # 1. กำหนดคำนำหน้าชื่อไฟล์ (Prefix) ตามทิศทาง
            if steering < -0.1:
                direction_str = "left"
            elif steering > 0.1:
                direction_str = "right"
            else:
                direction_str = "forward"
                
            # 2. สร้าง Timestamp ระดับมิลลิวินาที (เช่น 1786428593300)
            timestamp_ms = int(time.time() * 1000)
            
            # 3. ประกอบชื่อไฟล์ เช่น left_1786428593300.jpg
            img_name = f"{direction_str}_{timestamp_ms}.jpg"
            img_full_path = os.path.join(img_dir, img_name)
            
            # 4. เซฟรูปลงโฟลเดอร์ และเขียนลง CSV
            cv2.imwrite(img_full_path, cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
            writer.writerow([img_name, steering, throttle])

        # แสดงภาพสดๆ เพื่อเช็กว่ากล้องไม่ดำแล้ว
        cv2.imshow("Collecting Data", cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)
        time.sleep(0.02)

finally:
    csv_file.close()
    env.close()
    cv2.destroyAllWindows()
    print(f"\nบันทึกข้อมูลเสร็จสิ้นที่โฟลเดอร์: {base_dataset_dir}")