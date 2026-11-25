"""
เชื่อมต่อ Unity (CarAgentManual.cs, พอร์ต 9092) สำหรับเก็บข้อมูล / DAgger / รันโมเดล

รูปแบบข้อมูลที่ Unity ส่งมาทุก step (little-endian):
  status (1 byte) | run time (float32, s) | lap time (float32, s) | frame length (int32) | JPEG

  run time = เวลาตั้งแต่ reset   lap time = เวลารอบปัจจุบัน (ถ้า event = lap_complete คือเวลาของรอบที่เพิ่งจบ)
  เวลาทั้งสองจับโดย Unity เอง ความละเอียด 0.02 s และปัดเป็นทศนิยม 2 ตำแหน่งตรงนี้

API (เหมือนเดิม):
  obs, info = env.reset()
  obs, reward, done, truncated, info = env.step([steering, throttle])
  info = {"event": "driving" | "checkpoint" | "lap_complete", "run_time": 12.34, "lap_time": 6.78}
  obs เป็นภาพ RGB ขนาด 100x120
"""

import socket
import struct

import cv2
import numpy as np

OBS_HEIGHT = 100
OBS_WIDTH = 120

STATUS_EVENTS = {
    0: "driving",
    2: "checkpoint",
    3: "lap_complete",
}


class UnityManualCarEnv:
    def __init__(self, host="127.0.0.1", port=9092):
        self.host = host
        self.port = port
        self.sock = None

    # ------------------------------------------------------------------
    def _connect(self):
        if self.sock is None:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.sock.settimeout(10)
            self.sock.connect((self.host, self.port))
            print(f"[Unity] Connected to {self.host}:{self.port}")

    def _recvall(self, count):
        data = bytearray()

        while len(data) < count:
            packet = self.sock.recv(count - len(data))

            if not packet:
                raise ConnectionError("[Unity] Connection closed")

            data.extend(packet)

        return bytes(data)

    def _receive_state(self):
        status = self._recvall(1)[0]
        run_time, lap_time = struct.unpack("<ff", self._recvall(8))
        frame_len = struct.unpack("<I", self._recvall(4))[0]

        if frame_len == 0:
            frame = np.zeros((OBS_HEIGHT, OBS_WIDTH, 3), dtype=np.uint8)
        else:
            jpg = self._recvall(frame_len)
            bgr = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)

            if bgr is None:
                frame = np.zeros((OBS_HEIGHT, OBS_WIDTH, 3), dtype=np.uint8)
            else:
                if bgr.shape[:2] != (OBS_HEIGHT, OBS_WIDTH):
                    bgr = cv2.resize(bgr, (OBS_WIDTH, OBS_HEIGHT), interpolation=cv2.INTER_AREA)
                frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        info = {
            "event": STATUS_EVENTS.get(status, "driving"),
            "run_time": round(run_time, 2),
            "lap_time": round(lap_time, 2),
        }
        return frame, info

    # ------------------------------------------------------------------
    def reset(self):
        self._connect()
        self.sock.sendall(b"RESET\n")
        frame, info = self._receive_state()
        return frame, info

    def step(self, action):
        steering = float(np.clip(action[0], -1.0, 1.0))
        throttle = float(np.clip(action[1], -1.0, 1.0))

        self.sock.sendall(f"{steering:.3f},{throttle:.3f}\n".encode("utf-8"))
        frame, info = self._receive_state()

        # ไม่ตัดจบรอบเอง (Unity ไม่มีสถานะชน และครบรอบแล้ววิ่งต่อ) จึง done/truncated = False เสมอ
        return frame, 0.0, False, False, info

    def close(self):
        if self.sock is not None:
            try:
                self.sock.close()
            finally:
                self.sock = None