# Hairtail Processing System / 带鱼加工系统

An integrated fish-processing workstation project built around Raspberry Pi, FastAPI, computer vision, and Arduino-based motion control.  
这是一个围绕树莓派、FastAPI、机器视觉和 Arduino 运动控制构建的带鱼自动加工工作站项目。

## Overview / 项目简介

This repository contains a closed-loop prototype for hairtail processing, including:  
本仓库整理的是一个带鱼加工闭环原型系统，主要包括：

- visual length measurement with YOLO  
  基于 YOLO 的带鱼视觉测长
- web control panels for mobile and desktop  
  面向手机端和电脑端的网页控制界面
- serial communication with Arduino  
  树莓派与 Arduino 的串口通信
- conveyor, stopper, screw, sorting-servo, and cutting workflow control  
  输送、挡停、丝杆、分拣舵机和切刀等执行机构的联动控制
- automatic task generation based on measured fish length  
  根据测得长度自动生成加工任务

## Repository Structure / 仓库结构

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

## Main Components / 主要组成

### Python backend / Python 后端

- `fish_v5_backend_step1.py`
  - FastAPI backend  
    FastAPI 后端服务
  - camera capture and YOLO inference  
    摄像头采集与 YOLO 推理
  - serial communication with Arduino  
    与 Arduino 的串口通信
  - measurement history, task generation, and automatic processing flow  
    测长记录、任务生成和自动加工流程控制

### Web UI / 网页控制界面

- `index.html`
  - mobile-friendly operation page  
    手机端操作页面
- `pc.html`
  - desktop control page  
    电脑端控制页面

### Arduino firmware / Arduino 固件

- `arduino/fish_cutting_uno_r4_v3/fish_cutting_uno_r4_v3.ino`
  - motor and stepper control  
    电机与步进机构控制
  - limit switch handling  
    限位开关处理
  - sorting servo control  
    分拣舵机控制
  - cutter workflow support  
    切刀动作流程支持
  - serial command interface for the Raspberry Pi backend  
    面向树莓派后端的串口指令接口

## Hardware / Software Stack / 硬件与软件环境

- Raspberry Pi / 树莓派
- Python + FastAPI
- Picamera2
- YOLO / Ultralytics
- Arduino
- conveyor / screw / cutter / sorting mechanism  
  输送、丝杆、切刀、分拣执行机构

## Quick Start / 快速开始

### 1. Install dependencies / 安装依赖

```bash
pip install -r requirements.txt
```

### 2. Prepare model files / 准备模型文件

Place the exported YOLO NCNN model directory at:  
将导出的 YOLO NCNN 模型目录放到：

```text
models/best_ncnn_model
```

You can also override the model path with an environment variable:  
也可以通过环境变量覆盖模型路径：

```bash
export FISH_MODEL_PATH=/path/to/best_ncnn_model
```

### 3. Optional voice assets / 可选语音资源

Put broadcast audio files in:  
将播报语音文件放在：

```text
voice/
```

Or override with:  
或者通过环境变量指定：

```bash
export FISH_VOICE_AUDIO_DIR=/path/to/voice
```

### 4. Start backend / 启动后端

```bash
python fish_v5_backend_step1.py
```

Then open / 然后访问：

- `http://<device-ip>:8000/` for mobile UI / 手机端页面
- `http://<device-ip>:8000/pc` for desktop UI / 电脑端页面

### 5. Flash Arduino firmware / 烧录 Arduino 固件

Open and upload / 打开并烧录：

```text
arduino/fish_cutting_uno_r4_v3/fish_cutting_uno_r4_v3.ino
```

## Notes / 说明

- The backend automatically searches serial devices under `/dev/ttyACM*` and `/dev/ttyUSB*`.  
  后端会自动搜索 `/dev/ttyACM*` 和 `/dev/ttyUSB*` 下的串口设备。
- Runtime files such as `config.json`, `run_log.txt`, and snapshots are intentionally ignored from version control.  
  运行期生成的 `config.json`、`run_log.txt` 和抓拍图片等文件默认不纳入版本控制。
- This repository focuses on engineering implementation and deployment, so some parameters should be calibrated on the actual machine.  
  这个仓库更偏向工程实现与实际部署，因此部分参数需要按实机重新标定。
