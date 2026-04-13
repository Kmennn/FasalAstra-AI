# 🌾 FasalAstra AI
### Predictive Kinematic Weed Strike System — Raspberry Pi 4 Edition

> **Precision herbicide spraying for Indian smallholder farms.**  
> A YOLO-based computer vision system that detects weeds in real-time and fires a solenoid nozzle at the exact millisecond the weed enters the spray zone — saving up to 87% of herbicide vs. broadcast spraying.

---

## 📋 Table of Contents

1. [What Is This?](#what-is-this)
2. [How It Works — The Big Picture](#how-it-works)
3. [Model Performance](#model-performance)
4. [Project Structure](#project-structure)
5. [Hardware Required](#hardware-required)
6. [Hardware Wiring](#hardware-wiring)
7. [Software Setup — Raspberry Pi 4](#software-setup)
8. [Running the System](#running-the-system)
9. [Training the Model (PC)](#training-the-model)
10. [Core Modules Explained](#core-modules-explained)
11. [Dataset Information](#dataset-information)
12. [Safety Systems](#safety-systems)
13. [Troubleshooting](#troubleshooting)

---

## What Is This?

FasalAstra is a hand-held precision weed sprayer for Indian farmers. Instead of spraying the entire field (which wastes 87% of herbicide and damages crops), the farmer walks through the field holding a wand. The system:

1. **Sees** weeds using an OV5647 Pi Camera pointing at the ground
2. **Calculates** exactly when the weed will reach the spray nozzle
3. **Fires** a solenoid valve for exactly 50ms — spraying only a tiny precise burst on the weed

The intelligence lives inside a **Raspberry Pi 4** running a trained YOLO AI model. No cloud required. Works in remote farms with zero internet.

---

## How It Works

### The Kinematic Formula — The Heart of FasalAstra

This is what separates FasalAstra from a simple "spray when you see a weed" system.

```
Ti = (D / V) - (inference_latency + mechanical_latency)
```

Where:
- `Ti` = Time to start the solenoid timer (milliseconds)
- `D` = Distance from detected weed to spray nozzle (meters)
- `V` = Current walking velocity from MPU6050 IMU (m/s)
- `inference_latency` = 33ms (time for YOLO to detect the weed)
- `mechanical_latency` = 7ms (time for solenoid valve to physically open)

**Example:**  
Farmer walks at 1.2 m/s, weed detected 30cm ahead of nozzle:
```
Ti = (0.30 / 1.2) - (0.033 + 0.007)
Ti = 250ms - 40ms
Ti = 210ms  ← solenoid timer is set to fire in exactly 210ms
```

Without the kinematic prediction, the spray would always arrive **40ms too late** and miss the weed entirely.

### Full Frame-by-Frame Pipeline

```
Pi Camera (30 FPS)
    ↓
Frame captured (640×640)
    ↓
YOLO11n ONNX (10MB) detects weeds/crops/soil
    ↓  conf > 0.75
Weed detected → WeedTracker checks if already fired on this weed
    ↓  new weed only
HomographyMapper converts pixel position → real-world distance (meters)
    ↓
KinematicCalculator computes Ti (time to fire)
    ↓
CropExclusionZone checks — is a crop plant too close? If yes, ABORT
    ↓  safe to spray
Background thread: sleep(Ti ms) → fire solenoid for 50ms
    ↓
Display overlay shows bounding boxes, confidence scores, shot count
```

---

## Model Performance

Trained on **14,175 labeled images** (12,756 train / 1,419 val) from 14 diverse datasets, 200 epochs on RTX 3050.

| Metric | Score |
|---|---|
| Overall mAP50 | **76.89%** |
| Precision | 76.27% |
| Recall | 70.92% |
| **Weed mAP50** | **67.80%** ✅ |
| **Crop mAP50** | **85.97%** ✅ |
| Model size (ONNX) | 10 MB |
| Inference speed | ~33ms / frame |

> **Key Safety Metric:**  
> Crop mAP50 of **85.97%** means the model reliably identifies crops and avoids spraying them — protecting the farmer's yield.

---

## Project Structure

```
FasalAstra_AI/
│
├── models/
│   └── best.onnx           ← 🎯 DEPLOYED MODEL (10MB, use this on RPi4)
│
├── core/                   ← AI Pipeline Modules
│   ├── detector.py         ← YOLO inference wrapper
│   ├── kinematic.py        ← Ti formula calculator
│   ├── tracker.py          ← Anti-double-trigger system
│   ├── homography.py       ← Pixel → real-world distance mapper
│   └── exclusion.py        ← Crop safe-zone enforcer
│
├── rpi/                    ← Raspberry Pi Hardware Modules
│   ├── main_rpi.py         ← 🚀 MAIN ENTRY POINT — run this on RPi4
│   ├── gpio_controller.py  ← Solenoid valve controller (GPIO 17)
│   ├── imu_reader.py       ← MPU6050 velocity reader (I2C)
│   ├── field_calibrate.py  ← Camera-to-ground calibration tool
│   └── setup_rpi.sh        ← One-shot RPi4 dependency installer
│
├── datasets/               ← Training data (not on GitHub, 6.7GB)
│   └── merged/             ← Final merged dataset used for training
│       ├── images/train/
│       ├── images/val/
│       ├── labels/train/
│       ├── labels/val/
│       └── data.yaml
│
├── runs/                   ← Training outputs (not on GitHub)
│   └── detect/runs/fasal_astra_rpi4/
│       └── weights/
│           ├── best.pt     ← Full PyTorch weights
│           └── best.onnx   ← Exported lightweight ONNX
│
├── 01_download.py          ← Downloads all 14 datasets (Roboflow + Kaggle)
├── 02_rebuild_merge.py     ← Merges, validates, deduplicates all datasets
├── 03_train.py             ← Trains YOLO model on merged dataset
├── 04_evaluate.py          ← Evaluates model accuracy with readable report
├── 05_export.py            ← Exports best.pt → best.onnx for RPi4
├── 06_simulate.py          ← Software simulation (no hardware needed)
└── 08_results_visualizer.py← Visual dashboard of training results
```

---

## Hardware Required

| Component | Spec | Purpose |
|---|---|---|
| Raspberry Pi 4 | 4GB RAM model | Main compute unit |
| MicroSD Card | 32GB+ Class 10 | OS and code storage |
| Pi Camera OV5647 | 5MP CSI | Ground-facing vision |
| MPU6050 IMU | I2C, 6-axis | Wand velocity measurement |
| 5V Relay Module | Single channel, optocoupled | Safe GPIO→12V switching |
| 12V Solenoid Valve | Normally-closed, 1/4" NPT | Precision spray nozzle |
| 12V Battery | 2200mAh LiPo | Solenoid power |
| USB-C Power Bank | 5V 3A minimum | RPi4 power |
| HDMI Display (optional) | Any small HDMI | Live detection overlay |

---

## Hardware Wiring

### Solenoid via Relay (CRITICAL — read carefully)

> ⚠️ **Never connect the solenoid directly to GPIO pins.**  
> RPi GPIO is 3.3V / 16mA max. The solenoid runs at 12V / 500mA.  
> Direct connection will permanently destroy the Raspberry Pi.

```
RPi4 Pin 11 (GPIO 17) ──────────→ Relay IN
RPi4 Pin 1  (3.3V)    ──────────→ Relay VCC
RPi4 Pin 6  (GND)     ──────────→ Relay GND
Relay COM              ──────────→ 12V Battery (+)
Relay NO               ──────────→ Solenoid (+)
Solenoid (-)           ──────────→ 12V Battery (-)
```

### MPU6050 IMU via I2C

```
MPU6050 VCC ──→ RPi4 Pin 1  (3.3V)
MPU6050 GND ──→ RPi4 Pin 6  (GND)
MPU6050 SDA ──→ RPi4 Pin 3  (GPIO 2 / SDA)
MPU6050 SCL ──→ RPi4 Pin 5  (GPIO 3 / SCL)
```

### Pi Camera

```
Connect ribbon cable to RPi4 CSI port (labeled CAMERA)
Blue side of ribbon faces the Ethernet port
```

---

## Software Setup

### Step 1 — Flash Raspberry Pi OS

1. Download [Raspberry Pi Imager](https://www.raspberrypi.com/software/) on your PC
2. Open Imager → **Choose OS** → Raspberry Pi OS (64-bit)
3. **Choose Storage** → your MicroSD card
4. Click the **gear icon ⚙️** and configure:
   - Hostname: `fasalastra`
   - Username: `pi`, Password: `yourpassword`
   - Enable SSH ✅
   - Configure WiFi → your network name + password
5. Click **WRITE** → wait ~10 minutes

### Step 2 — Connect to the Pi

On your PC terminal:
```powershell
ssh pi@fasalastra.local
```
Type your password. Your prompt will change to `pi@fasalastra:~ $`

### Step 3 — Enable Interfaces

```bash
sudo raspi-config nonint do_camera 0
sudo raspi-config nonint do_i2c 0
```

### Step 4 — Install Dependencies

```bash
pip3 install ultralytics onnxruntime picamera2 RPi.GPIO smbus2 opencv-python numpy mpu6050-raspberrypi --break-system-packages
```

### Step 5 — Clone the Repository

```bash
git clone https://github.com/Kmennn/FasalAstra-AI.git
cd FasalAstra-AI
git checkout docs/final-comprehensive-reports
```

### Step 6 — Verify IMU is Connected (Optional)

```bash
sudo i2cdetect -y 1
```
You should see `68` in the grid output. That confirms the MPU6050 is connected correctly.

### Step 7 — Verify Camera is Connected (Optional)

```bash
libcamera-hello --timeout 2000
```
You should see a 2-second camera preview. If not, check the ribbon cable orientation.

---

## 🤖 Getting the Pre-Trained Model onto Raspberry Pi

> **This section is for teammates or anyone who just wants to run the system without re-training.**  
> The model is already trained and exported. You just need to get it onto the Pi — choose ONE method below.

---

### ✅ Method 1 — Git Clone (Easiest, Recommended)

This downloads the **entire project + model** in one command. Do this on the **Raspberry Pi terminal** after SSH-ing in.

```bash
# Clone the full project
git clone https://github.com/Kmennn/FasalAstra-AI.git

# Enter the project folder
cd FasalAstra-AI

# Switch to the branch that has the trained model
git checkout docs/final-comprehensive-reports

# Confirm the model is there (should show best.onnx ~10MB)
ls -lh models/
```

You should see something like:
```
-rw-r--r-- 1 pi pi 9.8M Apr 13 19:00 best.onnx
```

✅ Done! The model is at `~/FasalAstra-AI/models/best.onnx`.  
Now jump to [Running the System](#running-the-system).

---

### ✅ Method 2 — Direct Download (No Git needed)

If you just want the model file and nothing else, run this **on the Raspberry Pi**:

```bash
# Create the folder structure
mkdir -p ~/FasalAstra-AI/models

# Download the model directly from GitHub
wget -O ~/FasalAstra-AI/models/best.onnx \
  "https://github.com/Kmennn/FasalAstra-AI/raw/refs/heads/docs/final-comprehensive-reports/models/best.onnx"

# Then also download the Python scripts you need
git clone https://github.com/Kmennn/FasalAstra-AI.git /tmp/fasalastra
cp -r /tmp/fasalastra/core ~/FasalAstra-AI/
cp -r /tmp/fasalastra/rpi ~/FasalAstra-AI/
```

Verify the download:
```bash
ls -lh ~/FasalAstra-AI/models/best.onnx
# Should show: 9.8M best.onnx
```

---

### ✅ Method 3 — SCP from Teammate's Windows PC

If your teammate (who has the trained model on their Windows laptop) wants to push it directly to your Pi over WiFi:

**Run this on the teammate's Windows PC** (not the Pi):
```powershell
# First confirm both devices are on the same WiFi network
ping fasalastra.local

# Transfer the model file
scp "d:\THE DEVILS\FasalAstra_AI\models\best.onnx" pi@fasalastra.local:~/FasalAstra-AI/models/best.onnx
```

It will ask for the Pi password. Once done, the model is on the Pi. Then on the Pi, verify:
```bash
ls -lh ~/FasalAstra-AI/models/best.onnx
```

---

### After Getting the Model — Install Dependencies

Whichever method you used, you still need to install the Python libraries once:

```bash
pip3 install ultralytics onnxruntime picamera2 RPi.GPIO smbus2 opencv-python numpy mpu6050-raspberrypi --break-system-packages
```

Then run the system:
```bash
cd ~/FasalAstra-AI
python3 rpi/main_rpi.py
```

---

## Running the System

### Full Hardware Mode (on RPi4)

```bash
cd ~/FasalAstra-AI
python3 rpi/main_rpi.py
```

The system will:
1. Load `models/best.onnx` (10MB, ~3 second load time)
2. Initialize Pi Camera at 640×640
3. Initialize MPU6050 IMU at I2C address 0x68
4. Perform a 50ms solenoid test fire on startup
5. Begin real-time detection loop at ~30 FPS
6. Print live stats every frame:
   ```
   [Frame 142] FPS: 28.3 | Weeds: 2 | Fired: 5 | Aborted: 1 | Vel: 1.24 m/s
   ```

### Testing Without Hardware (Webcam Mode)

If you don't have the Pi Camera connected, open `rpi/main_rpi.py` and change:

```python
USE_PI_CAM = False   # line 36 — uses USB webcam instead
```

### Software Simulation Only (No Hardware at All)

```bash
python3 06_simulate.py
```

Runs a full simulation with synthetic weed positions and printed kinematic calculations. No camera, no GPIO, no hardware required. Great for verifying the code works before plugging in any hardware.

---

## Training the Model

> 💡 Only needed if you want to retrain or improve the model. The pre-trained `models/best.onnx` is ready to use.

### Requirements
- Windows/Linux PC with NVIDIA GPU (RTX 3050 or better)
- CUDA 11.8+
- Python 3.12
- Kaggle API token

### Step 1 — Set Up Environment

```powershell
python -m venv fasal_env
fasal_env\Scripts\activate
pip install ultralytics roboflow kaggle opencv-python
```

### Step 2 — Download Datasets

```powershell
python 01_download.py
```

Downloads ~6.7GB of labeled weed/crop images from:
- **Roboflow** (6 datasets): Augmented weeds, crop-weed mixed, canopy, cotton recognition, weed detection 2025
- **Kaggle** (3 datasets): DeepWeeds, crop-weed bounding boxes, plant seedlings
- **GitHub** (2 datasets): CWFID academic dataset, crop-weed field images
- **MH-Weed16**: 25,972 Indian farm weed images (manual download from Kaggle)

### Step 3 — Merge and Validate

```powershell
python 02_rebuild_merge.py
```

- Remaps all class labels to 3 unified classes: `weed (0)`, `crop (1)`, `soil (2)`
- Validates every image has a matching `.txt` label
- Removes corrupted files and duplicates
- Outputs to `datasets/merged/` — 14,175 clean images

### Step 4 — Train

```powershell
python 03_train.py
```

Trains YOLO11n for 200 epochs at 320×320 resolution on your GPU with batch size 64. Takes approximately 35-60 minutes on RTX 3050. Outputs to `runs/detect/runs/fasal_astra_rpi4/weights/best.pt`.

### Step 5 — Evaluate

```powershell
python 04_evaluate.py
```

Prints a clean accuracy report showing mAP50 per class and whether the model is safe to deploy.

### Step 6 — Export

```powershell
python 05_export.py
```

Converts `best.pt` → `best.onnx` (10MB) optimized for ARM CPU inference on the Raspberry Pi.

---

## Core Modules Explained

### `core/detector.py` — YOLO Inference

Wraps the YOLO model with a clean API. Confidence threshold is set to **0.75** (high precision — only fires on confident detections to prevent false positives on crops).

```python
detector = FasalAstraDetector('models/best.onnx')
detections = detector.detect(frame)
weeds = detector.get_weeds_only(detections)
```

### `core/kinematic.py` — Ti Calculator

Implements the core formula. Has 3 built-in safety checks:
- If farmer velocity < 0.1 m/s (stopped) → skip
- If weed > 0.55m from nozzle (out of range) → skip
- If Ti calculates negative (weed already passed) → skip

### `core/tracker.py` — Anti Double-Trigger

The camera runs at 30 FPS. The same weed appears in ~5-10 consecutive frames. Without this module, the system would fire the solenoid 5-10 times on a single weed — wasting enormous amounts of herbicide.

The tracker gives each weed a unique ID based on pixel position. Once a timer is set for a weed, it is **locked** for 2 seconds and ignored in all subsequent frames.

### `core/homography.py` — Pixel → Meters

Converts the pixel centroid of a detected weed bounding box into a real-world distance in meters (how far the weed is from the spray nozzle). Must be calibrated using `rpi/field_calibrate.py` when the camera angle or mounting height changes.

### `core/exclusion.py` — Crop Protection Zone

Defines a safety exclusion zone. If any crop bounding box overlaps or is within a configurable threshold of a weed, the spray is **aborted** even if the kinematic timer is already set. This is the last-resort crop protection layer.

### `rpi/gpio_controller.py` — Solenoid

Controls GPIO 17 via an optocoupled relay. The `fire(50)` call opens the solenoid valve for precisely 50ms. The relay provides electrical isolation — the 12V solenoid circuit is completely separated from the 3.3V Pi GPIO circuit.

### `rpi/imu_reader.py` — Velocity

Reads the MPU6050 6-axis accelerometer over I2C. Integrates the X-axis (forward motion) acceleration to compute velocity in m/s. Applies a low-pass filter (85% old velocity, 15% new measurement) to smooth out vibration noise. Falls back to 1.2 m/s default if IMU is disconnected.

---

## Dataset Information

### Class Mapping

| Class ID | Name | Training Samples |
|---|---|---|
| 0 | weed | ~28,500 bounding boxes |
| 1 | crop | ~17,900 bounding boxes |
| 2 | soil | Sparse (background class) |

### Data Sources

| Source | Images | Type |
|---|---|---|
| Roboflow rf_01_augmented_weeds | ~3,200 | Synthetic augmented |
| Roboflow rf_02_weedcrop | ~1,500 | Field images |
| Roboflow rf_03_crop_weed | ~2,100 | Indian farm |
| Roboflow rf_04_canopy | ~800 | Top-down view |
| Roboflow rf_05_mixed | ~1,200 | Mixed conditions |
| Kaggle kg_01_deepweeds | ~15,000 | Australian + Indian weeds |
| Kaggle kg_02_weed_detection | ~4,000 | Varied conditions |
| Kaggle kg_03_crop_weed_bbox | ~2,500 | Pre-labeled YOLO format |
| Kaggle kg_04_seedlings | ~5,500 | Early-stage plants |
| GitHub gh_01_cwfid | ~1,700 | Academic benchmark dataset |
| GitHub gh_02_crop_weed_github | ~3,000 | Community labeled |
| MH-Weed16 | ~25,972 | Indian field weeds (16 species) |

After validation and deduplication: **14,175 clean images** used for training.

---

## Safety Systems

FasalAstra has 5 independent safety layers to prevent spraying crops:

```
Layer 1: Confidence threshold 0.75      → ignores uncertain detections
Layer 2: Crop class detection          → model trained to identify crops
Layer 3: CropExclusionZone             → aborts if crop is near spray target
Layer 4: WeedTracker anti-double-fire  → prevents repeated firing
Layer 5: Velocity check                → won't fire if farmer is stationary
```

---

## Troubleshooting

### Camera not found
```bash
# Check camera is detected
libcamera-hello --timeout 2000
# If not detected, check ribbon cable (blue side faces Ethernet)
```

### IMU not found (0x68 missing)
```bash
# Verify I2C wiring
sudo i2cdetect -y 1
# Enable I2C if not already done
sudo raspi-config nonint do_i2c 0
# Then reboot
sudo reboot
```

### Model loads but no detections
- Check lighting — the model was trained on outdoor daytime images
- Clean the camera lens
- Adjust confidence threshold in `core/detector.py` (try lowering to `0.60` for testing)

### Solenoid not firing
- Confirm GPIO 17 → Relay IN connection
- Test relay manually:
  ```bash
  python3 -c "import RPi.GPIO as GPIO; GPIO.setmode(GPIO.BCM); GPIO.setup(17, GPIO.OUT); GPIO.output(17, GPIO.HIGH); import time; time.sleep(0.5); GPIO.output(17, GPIO.LOW); GPIO.cleanup()"
  ```
  You should hear the relay click.

### `picamera2` import error on PC
This is expected — `picamera2` only works on Raspberry Pi hardware. On a PC, set `USE_PI_CAM = False` in `main_rpi.py` to use a USB webcam instead.

---

## Quick Reference

| File | Command | Purpose |
|---|---|---|
| `rpi/main_rpi.py` | `python3 rpi/main_rpi.py` | **Run on RPi4 — live weed detection** |
| `06_simulate.py` | `python 06_simulate.py` | Test without any hardware |
| `04_evaluate.py` | `python 04_evaluate.py` | Check model accuracy |
| `03_train.py` | `python 03_train.py` | Retrain the model |
| `05_export.py` | `python 05_export.py` | Export .pt → .onnx |

---

*FasalAstra AI — Built for Indian farmers, powered by precision.*
