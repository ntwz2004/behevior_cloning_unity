import time
import cv2
import keyboard
from unity_env import UnityCarEnv

# เชื่อมต่อกับ Unity
env = UnityCarEnv()
obs, _ = env.reset()

print("========================================")
print("  ควบคุมรถด้วยคีย์บอร์ดแบบ Manual:")
print("  [ ลูกศร ขึ้น  ] : เดินหน้า")
print("  [ ลูกศร ลง   ] : ถอยหลัง / เบรก")
print("  [ ลูกศร ซ้าย  ] : เลี้ยวซ้าย")
print("  [ ลูกศร ขวา   ] : เลี้ยวขวา")
print("  (ไม่กดปุ่มเดินหน้า = รถจอดอยู่นิ่งๆ)")
print("  [ 'r'         ] : Reset รถกลับจุดเริ่มต้น")
print("  [ 'q'         ] : ออกจากโปรแกรม")
print("========================================")

try:
    while True:
        steering = 0.0
        throttle = 0.0

        # ควบคุมคันเร่ง (เดินหน้า / ถอยหลัง)
        if keyboard.is_pressed('up'):
            throttle = 1.0
        elif keyboard.is_pressed('down'):
            throttle = -0.6

        # ควบคุมการเลี้ยว
        if keyboard.is_pressed('left'):
            steering = -0.8
        elif keyboard.is_pressed('right'):
            steering = 0.8

        # สั่งรีเซ็ตรถ
        if keyboard.is_pressed('r'):
            obs, _ = env.reset()
            print("Resetting car...")
            time.sleep(0.2)
            continue

        # ออกจากโปรแกรม
        if keyboard.is_pressed('q'):
            break

        # ส่งทั้ง [steering, throttle] ไปให้ Unity
        obs, reward, done, truncated, _ = env.step([steering, throttle])

        # แสดงภาพกล้องหน้ารถ
        frame_bgr = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
        cv2.imshow("Agent Camera (Live)", frame_bgr)
        cv2.waitKey(1)

        # ชนแล้ว Reset ให้อัตโนมัติ
        if done:
            print("Collision / OffTrack! Resetting...")
            obs, _ = env.reset()
            time.sleep(0.3)

        time.sleep(0.02)

finally:
    env.close()
    cv2.destroyAllWindows()
    print("ปิดการเชื่อมต่อเรียบร้อย")