import os
import socket
import struct

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces


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

        # เซฟภาพ debug แค่ครั้งเดียวต่อการรัน Python
        self.debug_image_saved = False
        self.debug_image_path = os.path.abspath("ai_vision.jpg")

        # -1 = left, 0 = straight, +1 = right
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )

        # Unity ส่งภาพ RGB ขนาด 200 x 240
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(200, 240, 3),
            dtype=np.uint8,
        )

    def _connect(self):
        if self.sock is None:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(10)
            self.sock.connect((self.host, self.port))
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
        hit_data = self._recvall(1)

        if hit_data is None:
            print("[Unity] Cannot receive hit status")
            return np.zeros((200, 240, 3), dtype=np.uint8), 1

        frame_len_data = self._recvall(4)

        if frame_len_data is None:
            print("[Unity] Cannot receive frame size")
            return np.zeros((200, 240, 3), dtype=np.uint8), 1

        hit = int(hit_data[0])
        frame_len = struct.unpack("<I", frame_len_data)[0]

        if frame_len == 0:
            print("[Unity] Warning: received empty camera frame")
            return np.zeros((200, 240, 3), dtype=np.uint8), hit

        frame_data = self._recvall(frame_len)

        if frame_data is None:
            print("[Unity] Cannot receive complete camera frame")
            return np.zeros((200, 240, 3), dtype=np.uint8), 1

        frame = cv2.imdecode(
            np.frombuffer(frame_data, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )

        if frame is None:
            print("[Unity] Cannot decode JPEG camera frame")
            return np.zeros((200, 240, 3), dtype=np.uint8), 1

        # OpenCV BGR -> RGB สำหรับ CnnPolicy
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), hit

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

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._connect()
        self.steps = 0

        self.sock.sendall(b"RESET\n")
        observation, _ = self._receive_state()

        # เซฟเฉพาะภาพแรกของการรัน
        self._save_debug_image_once(observation)

        return observation, {}

    def step(self, action):
        steering = float(np.clip(action[0], -1.0, 1.0))
        throttle = 0.4
        self.steps += 1

        command = f"{steering:.3f},{throttle:.3f}\n".encode("utf-8")
        self.sock.sendall(command)

        observation, hit = self._receive_state()

        terminated = False
        truncated = False

        if hit == self.STATUS_CRASH:
            reward = -25.0
            terminated = True
            event = "crash"
        elif hit == self.STATUS_LAP_COMPLETE:
           
            reward = 100.0
            terminated = True
            event = "lap_complete"
        elif hit == self.STATUS_CHECKPOINT:
           
            reward = 35.0
            event = "checkpoint"
        else:
           
            reward = 0.1
            event = "driving"

        if terminated:
            print(f"[Episode End] {event.upper()} | step={self.steps}")

        elif event == "checkpoint":
            print(f"[Route] Correct checkpoint | step={self.steps}")

        if truncated:
            print(f"[Episode End] TIME LIMIT | step={self.steps}")

        return observation, reward, terminated, truncated, {"event": event}

    def close(self):
        if self.sock is not None:
            self.sock.close()
            self.sock = None
