# Autonomous Driving in Unity (BC / DAgger / RL)

โปรเจกต์ทดลองและเปรียบเทียบการขับรถอัตโนมัติในสภาพแวดล้อมจำลอง **Unity** ด้วย 3 แนวทางหลัก ได้แก่:

- **Behavior Cloning (BC)** — เรียนรู้เลียนแบบการขับจากชุดภาพที่มนุษย์เก็บด้วยโมเดล `NvidiaCNN` (PyTorch)
- **DAgger (Dataset Aggregation)** — ให้โมเดล BC ขับและให้มนุษย์กดแทรกแก้ไขพร้อมบันทึกภาพและสถิติลง Excel
- **Reinforcement Learning (PPO)** — เทรนเอเจนต์ด้วย PPO (`stable-baselines3`) พร้อมบันทึก TensorBoard, CSV และ Excel

---

## โครงสร้างโปรเจกต์

```text
behavior_clonning/
├── src/
│   ├── collect_bc.py          # เก็บข้อมูลภาพสำหรับ Behavior Cloning (Unity พอร์ต 9092)
│   ├── train_bc.py            # เทรนโมเดล NvidiaCNN จาก datasets/forward_obstacle
│   ├── drive_dagger.py        # รัน DAgger (AI ขับ + มนุษย์กดแทรก) พร้อมบันทึกสถิติ Excel
│   ├── dagger_log.py          # ตัวช่วยบันทึกสถิติการทำ DAgger ลง Excel
│   ├── train_rl.py            # เทรน PPO ตั้งแต่เริ่มต้น (Unity พอร์ต 9091)
│   ├── resume_train_rl.py     # โหลดโมเดล PPO และ VecNormalize เดิมมาเทรนต่อ
│   ├── train_log.py           # ตัวช่วยบันทึกสถิติการเทรน RL ลง CSV และ Excel
│   ├── drive_rl.py            # ทดสอบขับด้วยโมเดล PPO (พอร์ต 9091)
│   ├── drive.py               # สคริปต์กลางสำหรับให้โมเดล BC / DAgger / RL ขับใน Unity (พอร์ต 9092)
│   ├── model.py               # สถาปัตยกรรม NvidiaCNN (3 คลาส: forward, left, right)
│   ├── unity_env.py           # Gymnasium Environment สำหรับ RL (TCP พอร์ต 9091)
│   ├── unity_manual_env.py    # Environment สำหรับเก็บข้อมูล / DAgger / ทดสอบขับ (TCP พอร์ต 9092)
│   ├── train_datasets/
│   │   ├── dataset.py         # โหลดภาพ, ทำ Horizontal Flip และบาลานซ์คลาส
│   │   └── train.py           # สคริปต์เทรน BC จาก datasets/dataset_5
│   └── unity/
│       └── CarAgent.cs        # สคริปต์ฝั่ง Unity สำหรับเชื่อมต่อ TCP พอร์ต 9091
├── datasets/                  # โฟลเดอร์เก็บชุดข้อมูลภาพ (ไม่ถูก track ใน git)
├── models/                    # โฟลเดอร์เก็บโมเดลที่เทรนแล้ว (models/bc/, models/rl/)
├── logs/                      # โฟลเดอร์เก็บ log การเทรน, TensorBoard และ Excel
├── images/                    # ภาพประกอบสนามและภาพ debug (images/debug/ai_vision.jpg)
├── docs/                      # เอกสารประกอบโครงงาน
├── requirements.txt           # รายการไลบรารี Python ที่ต้องติดตั้ง
└── .gitignore
```

---

## การติดตั้ง

1. สร้างและเปิดใช้งาน Virtual Environment (แนะนำ Python 3.11):

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

2. ติดตั้งไลบรารีที่จำเป็น:

```powershell
pip install -r requirements.txt
```

> **หมายเหตุ (NVIDIA GPU):** หากต้องการใช้ CUDA ให้ติดตั้ง PyTorch ให้ตรงกับเวอร์ชัน CUDA ของเครื่อง เช่น
> ```powershell
> pip install torch --index-url https://download.pytorch.org/whl/cu124
> ```

---

## การเชื่อมต่อกับ Unity

ก่อนรันสคริปต์ฝั่ง Python ให้เปิดโปรเจกต์ใน **Unity Editor** และกด **Play** ก่อนทุกครั้ง โดยพอร์ต TCP ที่ใช้งานมี 2 ชุด:

| พอร์ต | ฝั่ง Unity | ฝั่ง Python | การใช้งาน |
|---:|---|---|---|
| `9091` | `CarAgent.cs` | `unity_env.py` | เทรนและรัน RL (`train_rl.py`, `resume_train_rl.py`, `drive_rl.py`) |
| `9092` | `CarAgentManual.cs` | `unity_manual_env.py` | เก็บข้อมูล BC, ทำ DAgger และทดสอบขับ (`collect_bc.py`, `drive_dagger.py`, `drive.py`) |

---

## วิธีใช้งาน

### 1. Behavior Cloning (BC)

**เก็บข้อมูลภาพ** (บันทึกลง `datasets/forward_obstacle/`):

```powershell
python src/collect_bc.py
```

- ปุ่มควบคุม: `↑` เดินหน้า | `←` เลี้ยวซ้าย | `→` เลี้ยวขวา | `R` รีเซ็ตรถ | `Q` ออก

**เทรนโมเดล NvidiaCNN** (เซฟโมเดลไว้ที่ `models/bc/`):

```powershell
python src/train_bc.py
```

*(หรือใช้ `python src/train_datasets/train.py` สำหรับเทรนจากชุดข้อมูล `datasets/dataset_5/`)*

---

### 2. DAgger

ให้โมเดล BC ขับและกดปุ่มแทรกเมื่อรถขับพลาด พร้อมบันทึกภาพใหม่ลง `datasets/dagger_unity_3/` และสรุปผลเป็น Excel ลง `logs/dagger_unity/`:

```powershell
python src/drive_dagger.py
```

- ก่อนรัน ตรวจสอบ path ของ `MODEL_FILE` ใน `src/drive_dagger.py` ให้ตรงกับไฟล์ `.pth` ใน `models/bc/`
- ปุ่มควบคุม: `↑` / `←` / `→` กดแทรกเพื่อสอนโมเดล | `↓` เบรก | `R` รีเซ็ตรถ | `Q` ออก

---

### 3. Reinforcement Learning (PPO)

**เริ่มเทรนใหม่** (เซฟโมเดลและ checkpoint ที่ `models/rl/` และ log ที่ `logs/rl/`):

```powershell
python src/train_rl.py
```

**เทรนต่อจากโมเดลเดิม** (โหลดโมเดลและไฟล์ `_vecnormalize.pkl` จาก `models/rl/` และเซฟที่ `models/rl/resumed/`):

```powershell
python src/resume_train_rl.py
```

*(แก้ไข `LOAD_MODEL_PATH` และ `LOAD_VECNORM_PATH` ใน `src/resume_train_rl.py` ให้ตรงกับไฟล์ที่มี)*

**ดูกราฟการเทรนด้วย TensorBoard:**

```powershell
tensorboard --logdir logs/rl
```

**ทดสอบขับด้วยโมเดล RL (พอร์ต 9091):**

```powershell
python src/drive_rl.py
```

---

### 4. ทดสอบขับและจับเวลาด้วย `drive.py` (รองรับทั้ง BC / DAgger / RL)

```powershell
# โมเดล BC หรือ DAgger (.pth)
python src/drive.py --type bc --model models/bc/nvidia_model_obstacle_5_2.pth
python src/drive.py --type dagger --model models/bc/model_unity_da_2.pth

# โมเดล RL (.zip)
python src/drive.py --type rl --model models/rl/ppo_car_lap_complete.zip
```

- ปุ่มควบคุม: `R` รีเซ็ตรถ | `Q` ออก
