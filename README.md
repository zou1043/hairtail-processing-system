# Hairtail Processing System

An integrated fish-processing workstation project built around Raspberry Pi, FastAPI, computer vision, and Arduino-based motion control.

## Overview

This repository contains a closed-loop prototype for hairtail processing, including:

- visual length measurement with YOLO
- web control panels for mobile and desktop
- serial communication with Arduino
- conveyor, stopper, screw, sorting-servo, and cutting workflow control
- automatic task generation based on measured fish length

## Repository Structure

```text
hairtail-processing-system/
├─ fish_v5_backend_step1.py
├─ index.html
├─ pc.html
├─ requirements.txt
├─ models/
├─ voice/
└─ arduino/
   └─ fish_cutting_uno_r4_v3/
      └─ fish_cutting_uno_r4_v3.ino
```

## Main Components

### Python backend

- `fish_v5_backend_step1.py`
  - FastAPI backend
  - camera capture and YOLO inference
  - serial communication with Arduino
  - measurement history, task generation, and automatic processing flow

### Web UI

- `index.html`
  - mobile-friendly operation page
- `pc.html`
  - desktop control page

### Arduino firmware

- `arduino/fish_cutting_uno_r4_v3/fish_cutting_uno_r4_v3.ino`
  - motor and stepper control
  - limit switch handling
  - sorting servo control
  - cutter workflow support
  - serial command interface for the Raspberry Pi backend

## Hardware / Software Stack

- Raspberry Pi
- Python + FastAPI
- Picamera2
- YOLO / Ultralytics
- Arduino
- conveyor / screw / cutting / sorting mechanism

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare model files

Place the exported YOLO NCNN model directory at:

```text
models/best_ncnn_model
```

You can also override the model path with an environment variable:

```bash
export FISH_MODEL_PATH=/path/to/best_ncnn_model
```

### 3. Optional voice assets

Put broadcast audio files in:

```text
voice/
```

Or override with:

```bash
export FISH_VOICE_AUDIO_DIR=/path/to/voice
```

### 4. Start backend

```bash
python fish_v5_backend_step1.py
```

Then open:

- `http://<device-ip>:8000/` for mobile UI
- `http://<device-ip>:8000/pc` for desktop UI

### 5. Flash Arduino firmware

Open and upload:

```text
arduino/fish_cutting_uno_r4_v3/fish_cutting_uno_r4_v3.ino
```

## Notes

- The backend automatically searches serial devices under `/dev/ttyACM*` and `/dev/ttyUSB*`.
- Runtime files such as `config.json`, `run_log.txt`, and snapshots are intentionally ignored from version control.
- This repository focuses on engineering implementation and deployment, so some parameters should be calibrated on the actual machine.
