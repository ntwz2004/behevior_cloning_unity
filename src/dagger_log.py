"""
บันทึกสถิติการทำ DAgger ลง Excel

การรันแต่ละครั้ง (session) สร้างไฟล์ Excel ใหม่แยกกัน ไม่เขียนทับหรือรวมกับ session ก่อนหน้า
ในแต่ละไฟล์มี 2 ชีต
- Summary  : สรุป session นี้ (จำนวนรูป, เวลาที่ใช้กดแทรก, ค่าตั้ง ฯลฯ)
- Overrides: 1 แถวต่อการกดแทรกหนึ่งช่วง (ปุ่มเดียวกันต่อเนื่อง = 1 ช่วง)

ต้องติดตั้ง: pip install openpyxl
"""

import glob
import os
import time
from datetime import datetime

try:
    from openpyxl import Workbook, load_workbook
    from openpyxl.styles import Font
except ImportError:  # รันต่อได้ แต่จะไม่มีไฟล์ Excel
    Workbook = None
    load_workbook = None
    Font = None

LABELS = ("forward", "left", "right")

SEGMENT_HEADERS = [
    "No.",
    "Label",
    "Start",
    "Duration (s)",
    "Images saved",
    "AI agreed (images)",
]

DATETIME_FORMAT = "yyyy-mm-dd hh:mm:ss.000"


def count_dataset_images(folder):
    """นับรูปทั้งหมดในโฟลเดอร์ แยกตาม label (ดูจากชื่อไฟล์ เช่น left_123.jpg)"""
    counts = {label: 0 for label in LABELS}

    for path in glob.glob(os.path.join(folder, "*.jpg")):
        label = os.path.basename(path).split("_", 1)[0]
        if label in counts:
            counts[label] += 1

    return counts


class OverrideTracker:
    """ติดตามช่วงเวลาที่ผู้ใช้กดแทรก (ปุ่มเดียวกันต่อเนื่อง = 1 ช่วง)"""

    def __init__(self):
        self.segments = []
        self.current = None

    def update(self, label, now=None):
        """เรียกทุกรอบ: label = ชื่อ label ที่กดอยู่ หรือ None ถ้าไม่ได้กด"""
        now = time.monotonic() if now is None else now

        if self.current is not None and self.current["label"] != label:
            self.close(now)

        if label is not None and self.current is None:
            self.current = {
                "label": label,
                "start_mono": now,
                "start_dt": datetime.now(),
                "images": 0,
                "agree": 0,
            }

    def add_image(self, agreed):
        if self.current is None:
            return

        self.current["images"] += 1
        if agreed:
            self.current["agree"] += 1

    def close(self, now=None):
        if self.current is None:
            return

        now = time.monotonic() if now is None else now
        segment = self.current
        segment["duration"] = now - segment["start_mono"]
        self.segments.append(segment)
        self.current = None

    def total_seconds(self, now=None):
        now = time.monotonic() if now is None else now
        total = sum(segment["duration"] for segment in self.segments)

        if self.current is not None:
            total += now - self.current["start_mono"]

        return total


def _style_header(ws):
    for cell in ws[1]:
        cell.font = Font(bold=True)

    for column in ws.columns:
        letter = column[0].column_letter
        width = max(len(str(column[0].value or "")) + 2, 12)
        ws.column_dimensions[letter].width = width


def save_excel(log_dir, session_name, folder, start_dt, end_dt, total_seconds, tracker,
               saved_count, agree_count, overruns, settings, dataset_before):
    """
    สร้างไฟล์ Excel ใหม่สำหรับ session นี้โดยเฉพาะ คืนค่า path ของไฟล์
    settings: dict ของค่าตั้งที่ต้องการบันทึก (model, base_speed, max_diff, ...)
    dataset_before: จำนวนรูปในโฟลเดอร์ก่อนเริ่ม session นี้
    overruns: จำนวนรอบที่ช้าเกินกำหนด (None = ไม่มีแนวคิดนี้ เช่นใน Unity)
    """
    if Workbook is None:
        print("\n[Excel] openpyxl is not installed: run 'pip install openpyxl'. Skipping Excel log.")
        return None

    tracker.close()

    os.makedirs(log_dir, exist_ok=True)
    stamp = start_dt.strftime("%Y%m%d_%H%M%S")
    file_name = f"dagger_{session_name}_{stamp}.xlsx" if session_name else f"dagger_{stamp}.xlsx"
    path = os.path.join(log_dir, file_name)

    segments = tracker.segments
    override_seconds = sum(segment["duration"] for segment in segments)
    override_pct = override_seconds / total_seconds * 100.0 if total_seconds > 0 else 0.0
    avg_press = override_seconds / len(segments) if segments else 0.0
    agree_pct = agree_count / saved_count * 100.0 if saved_count > 0 else 0.0

    saved_by_label = {label: 0 for label in LABELS}
    for segment in segments:
        saved_by_label[segment["label"]] += segment["images"]

    dataset_after = count_dataset_images(folder)

    wb = Workbook()
    summary = wb.active
    summary.title = "Summary"

    rows = [
        ("Metric", "Value"),
        ("Session name", session_name or "-"),
        ("Start", start_dt),
        ("End", end_dt),
        ("Total time (s)", round(total_seconds, 1)),
        ("Override time (s)", round(override_seconds, 1)),
        ("Override share of time (%)", round(override_pct, 1)),
        ("Override presses", len(segments)),
        ("Average press duration (s)", round(avg_press, 2)),
        ("Images saved (this session)", saved_count),
        ("  forward", saved_by_label["forward"]),
        ("  left", saved_by_label["left"]),
        ("  right", saved_by_label["right"]),
        ("AI agreed with human (%)", round(agree_pct, 1)),
        ("Images in dataset folder (before session)", dataset_before),
        ("Images in dataset folder (after session)", sum(dataset_after.values())),
    ]

    # แถวที่ขึ้นกับแพลตฟอร์ม: ใส่เฉพาะค่าที่ส่งมา (Pi ส่งครบ, Unity ไม่มี BASE_SPEED/MAX_DIFF/overruns)
    if overruns is not None:
        rows.append(("Loop overruns", overruns))

    optional = (
        ("Environment", "environment"),
        ("Model", "model"),
        ("BASE_SPEED", "base_speed"),
        ("MAX_DIFF", "max_diff"),
        ("THROTTLE", "throttle"),
        ("CONTROL_PERIOD", "control_period"),
        ("SAVE_MIN_INTERVAL", "save_min_interval"),
    )
    for title, key in optional:
        if settings.get(key) is not None:
            rows.append((title, settings[key]))

    for row in rows:
        summary.append(row)
        if isinstance(row[1], datetime):
            summary.cell(summary.max_row, 2).number_format = DATETIME_FORMAT

    _style_header(summary)
    summary.column_dimensions["A"].width = 44
    summary.column_dimensions["B"].width = 24

    overrides = wb.create_sheet("Overrides")
    overrides.append(SEGMENT_HEADERS)

    for number, segment in enumerate(segments, start=1):
        overrides.append([
            number,
            segment["label"],
            segment["start_dt"],
            round(segment["duration"], 2),
            segment["images"],
            segment["agree"],
        ])
        overrides.cell(overrides.max_row, 3).number_format = DATETIME_FORMAT

    _style_header(overrides)
    overrides.column_dimensions["C"].width = 26

    wb.save(path)
    return path