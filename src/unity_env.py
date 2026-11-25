import os
import socket
import struct

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces


OBS_HEIGHT = 100                # ต้องตรงกับ FrameHeight ฝั่ง Unity
OBS_WIDTH = 120                 # ต้องตรงกับ FrameWidth ฝั่ง Unity

# 1 step = fixedStepsPerAction physics step ใน Unity (ค่าเริ่มต้น 5 x 0.02s = 0.1 วินาที)

MAX_EPISODE_STEPS = 800         # ปรับลดลงตามความเหมาะสมเพื่อไม่ให้รถรอนานตอนติดขอบ
THROTTLE = 0.4

REWARD_DRIVING = 0.05           # reward พื้นฐานต่อ step
STEERING_PENALTY = 0.02         # โทษการหักพวงมาลัย
REWARD_CHECKPOINT = 15.0
REWARD_CRASH = -15.0
REWARD_LAP_COMPLETE = 50.0

VERBOSE_CHECKPOINT = False      # True = print ทุก checkpoint
VERBOSE_EPISODE_END = True      # True = print ตอนจบ episode


class UnityCarEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    STATUS_SAFE = 0
    STATUS_CRASH = 1
    STATUS_CHECKPOINT = 2
    STATUS_LAP_COMPLETE = 3

    def __init__(self, host="127.0.0.1", port=9091):
        super().__init__()

        self.host = host
        self.port = port
        self.steps = 0
        self.sock = None

        self.debug_image_saved = False
        project_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
        debug_images_dir = os.path.join(project_dir, "images", "debug")
        os.makedirs(debug_images_dir, exist_ok=True)
        self.debug_image_path = os.path.join(debug_images_dir, "ai_vision.jpg")

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(OBS_HEIGHT, OBS_WIDTH, 3),
            dtype=np.uint8,
        )

    def _blank_frame(self):
        return np.zeros((OBS_HEIGHT, OBS_WIDTH, 3), dtype=np.uint8)

    def _connect(self):
        if self.sock is None:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.settimeout(10)
            sock.connect((self.host, self.port))
            self.sock = sock
            print(f"[Unity] Connected to {self.host}:{self.port}")

    def _recvall(self, count):
        data = bytearray()
        while len(data) < count:
            packet = self.sock.recv(count - len(data))
            if not packet:
                return None
            data.extend(packet)
        return bytes(data)

    def _receive_state(self):
        # 1. รับ Status (1 byte)
        hit_data = self._recvall(1)
        if hit_data is None:
            raise ConnectionError("[Unity] Cannot receive hit status (connection closed)")
        hit = int(hit_data[0])

        # [อัปเดต] 2. รับ Raycast DistLeft และ DistRight (8 bytes: 2 floats)
        ray_data = self._recvall(8)
        if ray_data is None:
            raise ConnectionError("[Unity] Cannot receive raycast data")
        
        # "<ff" หมายถึงแกะแบบ Little-endian เป็น Float 32-bit จำนวน 2 ตัว
        dist_left, dist_right = struct.unpack("<ff", ray_data)

        # 3. รับขนาดความยาวข้อมูลภาพ (4 bytes)
        frame_len_data = self._recvall(4)
        if frame_len_data is None:
            raise ConnectionError("[Unity] Cannot receive frame size (connection closed)")
        frame_len = struct.unpack("<I", frame_len_data)[0]

        if frame_len == 0:
            print("[Unity] Warning: received empty camera frame")
            return self._blank_frame(), hit, dist_left, dist_right

        # 4. รับข้อมูลภาพ
        frame_data = self._recvall(frame_len)
        if frame_data is None:
            raise ConnectionError("[Unity] Cannot receive complete camera frame")

        frame = cv2.imdecode(
            np.frombuffer(frame_data, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )

        if frame is None:
            print("[Unity] Cannot decode JPEG camera frame")
            return self._blank_frame(), hit, dist_left, dist_right

        if frame.shape[:2] != (OBS_HEIGHT, OBS_WIDTH):
            frame = cv2.resize(
                frame, (OBS_WIDTH, OBS_HEIGHT), interpolation=cv2.INTER_AREA
            )

        # รีเทิร์นค่า dist_left และ dist_right เพิ่มกลับไปให้ฟังก์ชัน step()
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), hit, dist_left, dist_right

    def _save_debug_image_once(self, observation):
        if self.debug_image_saved:
            return

        image_bgr = cv2.cvtColor(observation, cv2.COLOR_RGB2BGR)
        saved = cv2.imwrite(self.debug_image_path, image_bgr)

        if saved:
            self.debug_image_saved = True
            print(f"[Debug] Saved camera image: {self.debug_image_path}")
        else:
            print(f"[Debug] Cannot save image: {self.debug_image_path}")

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._connect()
        self.steps = 0

        self.sock.sendall(b"RESET\n")
        
        # ดักรับ 4 ตัวแปร แต่เราไม่ได้ใช้ raycast ตอน reset จึงใส่ _ ทิ้งไว้
        observation, _, _, _ = self._receive_state()

        self._save_debug_image_once(observation)

        return observation, {}

    def step(self, action):
        steering = float(np.clip(action[0], -1.0, 1.0))
        self.steps += 1

        command = f"{steering:.3f},{THROTTLE:.3f}\n".encode("utf-8")
        self.sock.sendall(command)

        # [อัปเดต] รับค่าทั้งหมดจาก _receive_state
        observation, hit, dist_left, dist_right = self._receive_state()

        terminated = False
        truncated = False
        
        # ตัวแปรสำหรับเช็กเพื่อส่งออกไปเป็น log (ถ้ามี)
        lane_deviation = 0.0

        if hit == self.STATUS_CRASH:
            reward = REWARD_CRASH
            terminated = True
            event = "crash"

        elif hit == self.STATUS_LAP_COMPLETE:
            reward = REWARD_LAP_COMPLETE
            terminated = True
            event = "lap_complete"

        elif hit == self.STATUS_CHECKPOINT:
            reward = REWARD_CHECKPOINT
            event = "checkpoint"

        else:
            # [อัปเดต] การรวมค่าความห่างขอบสนามเข้าสมการ Reward
            lane_deviation = abs(dist_left - dist_right)
            
            # center_bonus: ยิ่งห่างจากศูนย์ ค่านี้ยิ่งลดลง (np.exp)
            # ตัวคูณ -1.5 ปรับเพิ่มลดได้ ถ้ารถแกว่งมากให้เพิ่มเป็น -2.0 หรือ -3.0
            center_bonus = np.exp(-1.5 * lane_deviation)
            
            # รวมเข้ากับ Reward หลัก
            # 0.5 * center_bonus คือน้ำหนักความสำคัญของการอยู่ตรงกลางเลน
            reward = REWARD_DRIVING + (0.5 * center_bonus) - (STEERING_PENALTY * abs(steering))
            event = "driving"

        if not terminated and self.steps >= MAX_EPISODE_STEPS:
            truncated = True
            event = "timeout"

        if VERBOSE_EPISODE_END:
            if terminated or truncated:
                print(f"[Episode End] {event.upper()} | step={self.steps} | reward={reward:.2f}")

        if VERBOSE_CHECKPOINT and event == "checkpoint":
            print(f"[Route] Correct checkpoint | step={self.steps} | reward={reward:.2f}")

        # พ่วง lane_deviation เข้าไปใน dict เพื่อใช้อ้างอิงหรือพล็อตกราฟใน TensorBoard
        info = {"event": event, "deviation": lane_deviation}

        return observation, reward, terminated, truncated, info

    def close(self):
        if self.sock is not None:
            try:
                self.sock.close()
            finally:
                self.sock = None
