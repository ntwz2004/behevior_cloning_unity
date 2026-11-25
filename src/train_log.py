"""
บันทึกสถิติการเทรน RL: จำนวน timestep, จำนวน episode และรายละเอียดทุก episode

- ระหว่างเทรน: เขียนทุก episode ลงไฟล์ CSV ทันที (ถ้าโปรแกรมถูก kill หรือไฟดับ ข้อมูลยังอยู่)
- ตอนเทรนจบ (ไม่ว่าจะจบแบบไหน): สร้างไฟล์ Excel ใหม่ 1 ไฟล์ต่อการเทรน 1 ครั้ง ไม่เขียนทับครั้งก่อน

ไฟล์ Excel มี 3 ชีต: Summary, Episodes, Settings
ต้องติดตั้ง: pip install openpyxl
"""

import csv
import os
import time
from datetime import datetime

try:
    from openpyxl import Workbook
    from openpyxl.styles import Font
except ImportError:  # เทรนต่อได้ แต่จะไม่มีไฟล์ Excel (ยังมี CSV)
    Workbook = None
    Font = None

EPISODE_HEADERS = [
    "Episode",
    "Timestep (end)",
    "Length (steps)",
    "Reward",
    "Reward (avg last 20)",
    "Checkpoints",
    "Mean lane deviation",
    "End event",
    "Time since start (s)",
]

DATETIME_FORMAT = "yyyy-mm-dd hh:mm:ss"


def _mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


class TrainingRecorder:
    def __init__(self, log_dir, run_name=""):
        self.start_dt = datetime.now()
        self.start_mono = time.monotonic()
        self.run_name = run_name
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        stamp = self.start_dt.strftime("%Y%m%d_%H%M%S")
        base = f"train_{run_name}_{stamp}" if run_name else f"train_{stamp}"
        self.csv_path = os.path.join(log_dir, base + "_episodes.csv")
        self.xlsx_path = os.path.join(log_dir, base + ".xlsx")

        self.rows = []
        self._csv_file = open(self.csv_path, "w", newline="", encoding="utf-8")
        self._csv = csv.writer(self._csv_file)
        self._csv.writerow(EPISODE_HEADERS)
        self._csv_file.flush()

    # ------------------------------------------------------------------
    def record_episode(self, timestep, length, reward, event, checkpoints, mean_deviation):
        """เรียกทุกครั้งที่จบ 1 episode"""
        recent = [row["reward"] for row in self.rows[-19:]] + [reward]

        row = {
            "episode": len(self.rows) + 1,
            "timestep": int(timestep),
            "length": int(length),
            "reward": float(reward),
            "reward_avg20": _mean(recent),
            "checkpoints": int(checkpoints),
            "deviation": float(mean_deviation),
            "event": event or "unknown",
            "elapsed": time.monotonic() - self.start_mono,
        }
        self.rows.append(row)

        self._csv.writerow([
            row["episode"],
            row["timestep"],
            row["length"],
            round(row["reward"], 3),
            round(row["reward_avg20"], 3),
            row["checkpoints"],
            round(row["deviation"], 4),
            row["event"],
            round(row["elapsed"], 1),
        ])
        self._csv_file.flush()

    def close_csv(self):
        try:
            self._csv_file.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    def _lap_stats(self):
        """นับรอบที่วิ่งครบ และจำนวนรอบติดกันที่ยาวที่สุด (ตรรกะเดียวกับ StopOnConsistentLapCallback)"""
        laps = 0
        streak = 0
        best_streak = 0
        first_lap = None

        for row in self.rows:
            if row["event"] == "lap_complete":
                laps += 1
                streak += 1
                best_streak = max(best_streak, streak)

                if first_lap is None:
                    first_lap = row
            elif row["event"] in ("crash", "timeout"):
                streak = 0

        return laps, best_streak, first_lap

    @staticmethod
    def _count_events(rows):
        counts = {}
        for row in rows:
            counts[row["event"]] = counts.get(row["event"], 0) + 1
        return counts

    # ------------------------------------------------------------------
    def save_excel(self, total_timesteps, stop_reason, settings, model_paths=None):
        """
        สร้างไฟล์ Excel ใหม่สำหรับการเทรนครั้งนี้ คืนค่า path ที่เซฟ (หรือ None)
        total_timesteps: model.num_timesteps ตอนจบ (รวม step ของ episode ที่ยังไม่จบด้วย)
        settings: dict ของค่าตั้ง (hyperparameters, reward, ...) เก็บลงชีต Settings
        """
        self.close_csv()

        if Workbook is None:
            print("\n[Excel] openpyxl is not installed: run 'pip install openpyxl'. "
                  f"Episode data is still in {self.csv_path}")
            return None

        end_dt = datetime.now()
        wall = time.monotonic() - self.start_mono
        rows = self.rows
        last100 = rows[-100:]

        steps_in_episodes = sum(row["length"] for row in rows)
        unfinished = max(0, int(total_timesteps) - steps_in_episodes)
        laps, best_streak, first_lap = self._lap_stats()
        events_all = self._count_events(rows)
        events_last = self._count_events(last100)
        n_steps = settings.get("n_steps")
        rollouts = int(total_timesteps) // int(n_steps) if n_steps else None

        wb = Workbook()
        summary = wb.active
        summary.title = "Summary"

        def pct(count, total):
            return round(count / total * 100.0, 1) if total else 0.0

        table = [
            ("Metric", "Value"),
            ("Run name", self.run_name or "-"),
            ("Start", self.start_dt),
            ("End", end_dt),
            ("Stopped because", stop_reason),
            ("Wall-clock time (h:mm:ss)", f"{int(wall // 3600)}:{int(wall % 3600 // 60):02d}:{int(wall % 60):02d}"),
            ("Wall-clock time (hours)", round(wall / 3600.0, 3)),
            ("", ""),
            ("Total timesteps", int(total_timesteps)),
            ("PPO rollouts (timesteps / n_steps)", rollouts if rollouts is not None else "-"),
            ("Timesteps per second", round(total_timesteps / wall, 1) if wall > 0 else 0),
            ("", ""),
            ("Episodes (finished)", len(rows)),
            ("Timesteps in finished episodes", steps_in_episodes),
            ("Timesteps in unfinished last episode", unfinished),
            ("", ""),
            ("Mean episode length (all)", round(_mean(r["length"] for r in rows), 1)),
            ("Mean episode length (last 100)", round(_mean(r["length"] for r in last100), 1)),
            ("Min / max episode length", f"{min((r['length'] for r in rows), default=0)} / "
                                         f"{max((r['length'] for r in rows), default=0)}"),
            ("Mean reward (all)", round(_mean(r["reward"] for r in rows), 2)),
            ("Mean reward (last 100)", round(_mean(r["reward"] for r in last100), 2)),
            ("Best episode reward", round(max((r["reward"] for r in rows), default=0.0), 2)),
            ("", ""),
            ("Laps completed (total)", laps),
            ("Longest streak of consecutive laps", best_streak),
            ("First lap: episode", first_lap["episode"] if first_lap else "-"),
            ("First lap: timestep", first_lap["timestep"] if first_lap else "-"),
            ("", ""),
            ("Episodes ended by crash", events_all.get("crash", 0)),
            ("Episodes ended by timeout", events_all.get("timeout", 0)),
            ("Episodes ended by lap_complete", events_all.get("lap_complete", 0)),
            ("Crash rate (all, %)", pct(events_all.get("crash", 0), len(rows))),
            ("Crash rate (last 100, %)", pct(events_last.get("crash", 0), len(last100))),
            ("Lap rate (last 100, %)", pct(events_last.get("lap_complete", 0), len(last100))),
            ("", ""),
            ("Mean checkpoints per episode (all)", round(_mean(r["checkpoints"] for r in rows), 2)),
            ("Mean checkpoints per episode (last 100)", round(_mean(r["checkpoints"] for r in last100), 2)),
        ]

        for row in table:
            summary.append(row)
            if isinstance(row[1], datetime):
                summary.cell(summary.max_row, 2).number_format = DATETIME_FORMAT

        summary.append(("", ""))
        summary.append(("Files", ""))
        summary.append(("Episode CSV (written live)", os.path.abspath(self.csv_path)))
        for name, path in (model_paths or {}).items():
            summary.append((name, os.path.abspath(path)))

        for cell in summary[1]:
            cell.font = Font(bold=True)
        summary.cell(summary.max_row - len(model_paths or {}) - 1, 1).font = Font(bold=True)
        summary.column_dimensions["A"].width = 44
        summary.column_dimensions["B"].width = 60

        # ---- Episodes ----
        sheet = wb.create_sheet("Episodes")
        sheet.append(EPISODE_HEADERS)
        for row in rows:
            sheet.append([
                row["episode"],
                row["timestep"],
                row["length"],
                round(row["reward"], 3),
                round(row["reward_avg20"], 3),
                row["checkpoints"],
                round(row["deviation"], 4),
                row["event"],
                round(row["elapsed"], 1),
            ])

        for cell in sheet[1]:
            cell.font = Font(bold=True)
        for letter, width in zip("ABCDEFGHI", (10, 16, 16, 12, 20, 13, 20, 14, 20)):
            sheet.column_dimensions[letter].width = width
        sheet.freeze_panes = "A2"

        # ---- Settings ----
        cfg = wb.create_sheet("Settings")
        cfg.append(("Setting", "Value"))
        for key, value in settings.items():
            cfg.append((key, value))
        for cell in cfg[1]:
            cell.font = Font(bold=True)
        cfg.column_dimensions["A"].width = 30
        cfg.column_dimensions["B"].width = 30

        try:
            wb.save(self.xlsx_path)
            return self.xlsx_path
        except PermissionError:
            # ไฟล์ถูกเปิดค้างอยู่ในโปรแกรมอื่น: เซฟเป็นไฟล์สำรองแทน ไม่ให้ข้อมูลหาย
            backup = self.xlsx_path.replace(".xlsx", f"_backup_{int(time.time())}.xlsx")
            wb.save(backup)
            return backup