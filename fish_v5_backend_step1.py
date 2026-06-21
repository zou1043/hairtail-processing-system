from collections import deque
from dataclasses import dataclass, field
import asyncio
import gc
import glob
import json
import os
import shutil
import subprocess
import threading
import time
from queue import Empty, Queue
from typing import Any, Dict, List, Optional

import cv2
from fastapi import APIRouter, Body, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
import numpy as np
import psutil
import serial
from picamera2 import Picamera2
from ultralytics import YOLO
import uvicorn

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))

# =========================================================
# Core configuration (Raspberry Pi 5 / 4GB tuned)
# =========================================================
MODEL_PATH = os.environ.get(
    "FISH_MODEL_PATH",
    os.path.join(BACKEND_DIR, "models", "best_ncnn_model"),
)
CAMERA_SIZE = (1640, 1232)
DISPLAY_SIZE = (640, 480)
CAMERA_FPS = 30
DETECTION_CONF = 0.3
PIXELS_PER_CM = 30.0
BAUDRATE = 115200
YOLO_HEAD_CLASS_ID = 1
YOLO_BODY_CLASS_ID = 0
YOLO_MEASURE_CLASS_ID = 2
MEASURE_REAL_LENGTH_MM = 150.0
DEFAULT_MM_PER_PIXEL = 10.0 / PIXELS_PER_CM
LIVE_OVERLAY_MAX_FPS = 8.0

DEFAULT_FEED_SPEED = 70
DEFAULT_STABLE_FRAMES = 3
DEFAULT_TOLERANCE = 1.0
DEFAULT_COOLDOWN = 2.0
DEFAULT_ABSENT_FRAMES = 8
DEFAULT_RESUME_DELAY = 0.5
DEFAULT_SETTLE_MS = 200
DEFAULT_SORT_HOLD_MS = 600
DEFAULT_CENTER_HOLD_MS = 250
DEFAULT_EJECT_MS = 2000
DEFAULT_CUT_MOTOR_SPEED = 85
DEFAULT_PUMP_CUT_SPEED = 0
DEFAULT_CUT_TIME_MS = 1200
DEFAULT_BLADE_OFFSET_MM = 45.0
DEFAULT_SERVO_CENTER = 90
DEFAULT_SERVO_HEAD = 0
DEFAULT_SERVO_BODY = 180
DEFAULT_SERVO2_DOWN = 0
DEFAULT_SERVO2_UP = 180
DEFAULT_VOICE_BROADCAST = True
VOICE_AUDIO_DIR = os.environ.get(
    "FISH_VOICE_AUDIO_DIR",
    os.path.join(BACKEND_DIR, "voice"),
)

MAX_MEASUREMENT_ROWS = 8
SNAPSHOT_QUALITY = 92
SNAPSHOT_DIR = os.path.expanduser("~/fish_workstation_snapshots")
BLANK_IMAGE = "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///ywAAAAAAQABAAACAUwAOw=="
ARDUINO_MAX_TASKS = 24
STEPPER_SOFT_LIMIT_MM = 80.0


def _build_blank_snapshot_jpeg() -> bytes:
    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
    success, jpeg = cv2.imencode(".jpg", placeholder, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return jpeg.tobytes() if success else b""


LATEST_SNAPSHOT_PLACEHOLDER = _build_blank_snapshot_jpeg()

# 手动高精度连拍
BURST_CAPTURE_COUNT = 5
BURST_CAPTURE_INTERVAL = 0.12

# 自动模式低频检测（树莓派 5 4GB 建议）
AUTO_DETECT_INTERVAL = 0.55
AUTO_DETECT_BURST_COUNT = 3
AUTO_DETECT_BURST_INTERVAL = 0.06

# Mobile/websocket stability tuning
MOBILE_UI_REFRESH_INTERVAL = 1.8
DISCONNECT_PROTECT_GRACE_SECONDS = 10.0

CONFIG_FILE_PATH = os.path.join(BACKEND_DIR, "config.json")
RUN_LOG_FILE_PATH = os.path.join(BACKEND_DIR, "run_log.txt")
LOG_HISTORY_LIMIT = 50

CONFIG_INT_FIELDS = {
    "fixed_length_mm",
    "avg_parts_count",
    "auto_speed",
    "stable_frames",
    "absent_frames",
    "settle_ms",
    "sort_hold_ms",
    "center_hold_ms",
    "eject_ms",
    "cut_time_ms",
    "servo_center",
    "servo_head",
    "servo_body",
    "servo2_down",
    "servo2_up",
    "cut_motor_speed",
    "pump_cut_speed",
}
CONFIG_FLOAT_FIELDS = {
    "tolerance",
    "cooldown",
    "step_dist_mm",
    "blade_offset_mm",
}
CONFIG_BOOL_FIELDS = {"voice_broadcast", "auto_enabled"}
CONFIG_STR_FIELDS = {"cut_mode"}
PERSISTED_CONFIG_FIELDS = (
    "cut_mode",
    "voice_broadcast",
    "fixed_length_mm",
    "avg_parts_count",
    "auto_speed",
    "stable_frames",
    "absent_frames",
    "tolerance",
    "cooldown",
    "settle_ms",
    "sort_hold_ms",
    "center_hold_ms",
    "eject_ms",
    "cut_time_ms",
    "servo_center",
    "servo_head",
    "servo_body",
    "servo2_down",
    "servo2_up",
    "cut_motor_speed",
    "pump_cut_speed",
    "blade_offset_mm",
    "step_dist_mm",
    "auto_enabled",
)
DEFAULT_CONFIG = {
    "cut_mode": "fixed",
    "voice_broadcast": DEFAULT_VOICE_BROADCAST,
    "fixed_length_mm": 60,
    "avg_parts_count": 4,
    "auto_speed": DEFAULT_FEED_SPEED,
    "stable_frames": DEFAULT_STABLE_FRAMES,
    "absent_frames": DEFAULT_ABSENT_FRAMES,
    "tolerance": DEFAULT_TOLERANCE,
    "cooldown": DEFAULT_COOLDOWN,
    "settle_ms": DEFAULT_SETTLE_MS,
    "sort_hold_ms": DEFAULT_SORT_HOLD_MS,
    "center_hold_ms": DEFAULT_CENTER_HOLD_MS,
    "eject_ms": DEFAULT_EJECT_MS,
    "cut_time_ms": DEFAULT_CUT_TIME_MS,
    "servo_center": DEFAULT_SERVO_CENTER,
    "servo_head": DEFAULT_SERVO_HEAD,
    "servo_body": DEFAULT_SERVO_BODY,
    "servo2_down": DEFAULT_SERVO2_DOWN,
    "servo2_up": DEFAULT_SERVO2_UP,
    "cut_motor_speed": DEFAULT_CUT_MOTOR_SPEED,
    "pump_cut_speed": DEFAULT_PUMP_CUT_SPEED,
    "blade_offset_mm": DEFAULT_BLADE_OFFSET_MM,
    "step_dist_mm": 10.0,
    "auto_enabled": False,
}


# =========================================================
# Serial communication
# =========================================================
class SerialManager:
    def __init__(self, pos_callback=None, log_callback=None, state_callback=None, line_callback=None):
        self.pos_callback = pos_callback
        self.log_callback = log_callback
        self.state_callback = state_callback
        self.line_callback = line_callback

        self.ser = None
        self.port = None
        self.running = False
        self.connected = False
        self.current_pos_mm = 0.0
        self._serial_lock = threading.Lock()
        self._line_condition = threading.Condition()
        self._line_seq = 0
        self._recent_lines = deque(maxlen=200)
        self._suppress_weight_log_count = 0
        self._startup_event = threading.Event()

        self.connect()

    def find_port(self):
        ports = sorted(glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*"))
        return ports[0] if ports else None

    def _log(self, msg):
        if self.log_callback:
            self.log_callback(msg)

    def _notify_state(self):
        if self.state_callback:
            self.state_callback(self.connected, self.port)

    def connect(self, silent=False):
        self.close(log=False)

        self.port = self.find_port()
        if not self.port:
            self.connected = False
            self._notify_state()
            if not silent:
                self._log("未检测到 Arduino，请检查 USB 连接。")
            return False

        try:
            self.ser = serial.Serial(self.port, BAUDRATE, timeout=0.1)
            self._startup_event.clear()
            time.sleep(1.8)
            try:
                self.ser.reset_input_buffer()
                self.ser.reset_output_buffer()
            except Exception:
                pass
            self.running = True
            self.connected = True
            threading.Thread(target=self._read_loop, daemon=True).start()
            startup_deadline = time.time() + 3.0
            while time.time() < startup_deadline and self.running and self.ser and self.ser.is_open:
                if self._startup_event.wait(timeout=0.2):
                    break
            self._notify_state()
            if not silent:
                self._log(f"硬件连接成功：{self.port}")
            return True
        except Exception as e:
            self.ser = None
            self.running = False
            self.connected = False
            self._notify_state()
            if not silent:
                self._log(f"串口打开失败：{e}")
            return False

    def reconnect(self, silent=False):
        if not silent:
            self._log("正在重新连接串口...")
        return self.connect(silent=silent)

    def _await_response(self, start_seq, wait_prefixes, reject_prefixes, timeout):
        deadline = time.time() + timeout
        with self._line_condition:
            while True:
                for seq, line in list(self._recent_lines):
                    if seq <= start_seq:
                        continue
                    if reject_prefixes and any(line.startswith(prefix) for prefix in reject_prefixes):
                        return False, line
                    if wait_prefixes and any(line.startswith(prefix) for prefix in wait_prefixes):
                        return True, line

                remaining = deadline - time.time()
                if remaining <= 0:
                    return None, None
                self._line_condition.wait(timeout=remaining)

    def _handle_serial_failure(self, log_message=None):
        if log_message:
            self._log(log_message)

        self.running = False
        self.connected = False

        ser = self.ser
        self.ser = None

        if ser:
            try:
                if ser.is_open:
                    ser.close()
            except Exception:
                pass

        self._notify_state()

    def send_cmd(self, cmd: str, wait_prefixes=None, reject_prefixes=None, timeout=2.5):
        if not (self.ser and self.ser.is_open):
            self._log(f"串口未连接，命令未发送：{cmd}")
            return False

        try:
            with self._serial_lock:
                with self._line_condition:
                    start_seq = self._line_seq
                self.ser.write((cmd + "\n").encode())

            if not wait_prefixes:
                return True

            matched, line = self._await_response(
                start_seq=start_seq,
                wait_prefixes=tuple(wait_prefixes),
                reject_prefixes=tuple(reject_prefixes or ("ERR ", "JOBERROR", "STOPPED")),
                timeout=timeout,
            )
            if matched is True:
                return True
            if matched is False:
                self._log(f"命令执行被拒绝：{cmd} -> {line}")
                return False

            self._log(f"命令执行超时：{cmd}")
            return False
        except Exception as e:
            self._log(f"命令发送失败：{e}")
            self._handle_serial_failure()
            return False

    def send_cmd_get_line(self, cmd: str, wait_prefixes, reject_prefixes=None, timeout=2.5, silent=False):
        if not (self.ser and self.ser.is_open):
            if not silent:
                self._log(f"串口未连接，命令未发送：{cmd}")
            return None

        try:
            with self._serial_lock:
                with self._line_condition:
                    start_seq = self._line_seq
                self.ser.write((cmd + "\n").encode())

            matched, line = self._await_response(
                start_seq=start_seq,
                wait_prefixes=tuple(wait_prefixes),
                reject_prefixes=tuple(reject_prefixes or ("ERR ", "JOBERROR", "STOPPED")),
                timeout=timeout,
            )
            if matched is True:
                return line
            if matched is False:
                if not silent:
                    self._log(f"命令执行被拒绝：{cmd} -> {line}")
                return None

            if not silent:
                self._log(f"命令执行超时：{cmd}")
            return None
        except Exception as e:
            if not silent:
                self._log(f"命令发送失败：{e}")
            self._handle_serial_failure()
            return None

    def _read_loop(self):
        while self.running and self.ser:
            try:
                raw = self.ser.readline()
                if not raw:
                    continue

                line = raw.decode("utf-8", errors="ignore").strip()
                if not line:
                    continue

                with self._line_condition:
                    self._line_seq += 1
                    self._recent_lines.append((self._line_seq, line))
                    self._line_condition.notify_all()

                if line.startswith("UNO R4 READY") or line.startswith("LIMITS"):
                    self._startup_event.set()

                if "POS=" in line:
                    try:
                        pos_text = line.split("POS=")[-1].strip()
                        self.current_pos_mm = float(pos_text)
                        if self.pos_callback:
                            self.pos_callback(self.current_pos_mm)
                    except Exception:
                        pass

                if self.line_callback:
                    self.line_callback(line)

                if self._suppress_weight_log_count > 0 and (
                    line.startswith("CMD: WEIGHT")
                    or line.startswith("WEIGHT=")
                    or line.startswith("WEIGHT_ERR")
                ):
                    self._suppress_weight_log_count -= 1
                    continue

                self._log(f"Arduino: {line}")
            except Exception as e:
                if self.running:
                    self._log(f"串口读取异常：{e}")
                    self._handle_serial_failure()
                break

        if self.running:
            self._handle_serial_failure()

    def close(self, log=True):
        self.running = False

        if self.ser:
            try:
                if self.ser.is_open:
                    try:
                        self.ser.write(b"STOP\n")
                    except Exception:
                        pass
                    self.ser.close()
            except Exception as e:
                self._log(f"串口关闭异常：{e}")

        was_connected = self.connected
        self.ser = None
        self.connected = False
        self.port = None
        self._notify_state()

        if log and was_connected:
            self._log("串口已关闭。")


# =========================================================
# Vision controller
# =========================================================
class VisionController:
    def __init__(self, ui_handlers):
        self.h = ui_handlers
        self.picam2 = None
        self.model = None
        self.running = False
        self.vision_enabled = False
        self.detection_paused = False
        self.pixels_per_cm = PIXELS_PER_CM
        self.mm_per_pixel = DEFAULT_MM_PER_PIXEL
        self.last_raw_frame = None
        self.last_frame_captured_at = 0.0
        self.latest_jpeg = None
        self.pending_status = None

        self.frame_queue = Queue(maxsize=1)
        self.display_queue = Queue(maxsize=1)
        self._state_lock = threading.Lock()
        self._run_token = 0

        self._frame_lock = threading.Lock()
        self._snapshot_lock = threading.Lock()
        self._auto_detect_lock = threading.Lock()
        self._model_lock = threading.Lock()

    def ensure_model(self):
        if self.model is None:
            self.model = YOLO(MODEL_PATH, task="segment")

    def _run_model(self, frame):
        self.ensure_model()
        if self.model is None:
            return []
        with self._model_lock:
            return self.model(frame, imgsz=640, verbose=False)

    def _robust_average(self, values):
        if not values:
            return 0.0
        ordered = sorted(float(v) for v in values if v > 0)
        if not ordered:
            return 0.0
        fused = ordered[1:-1] if len(ordered) >= 3 else ordered
        return sum(fused) / len(fused)

    def _mask_points_from_array(self, mask_array, frame_shape):
        mask = mask_array.astype(np.float32)
        if mask.shape != frame_shape[:2]:
            mask = cv2.resize(mask, (frame_shape[1], frame_shape[0]), interpolation=cv2.INTER_NEAREST)
        binary_mask = mask > 0.5
        ys, xs = np.where(binary_mask)
        if xs.size < 5:
            return None, None
        points = np.column_stack((xs.astype(np.float32), ys.astype(np.float32)))
        return binary_mask, points

    def _apply_mask_overlay(self, annotated, binary_mask, polygon, color):
        overlay = annotated.copy()
        if polygon is not None and len(polygon) >= 3:
            cv2.fillPoly(overlay, [polygon], color)
            cv2.addWeighted(overlay, 0.30, annotated, 0.70, 0, annotated)
            cv2.polylines(annotated, [polygon], True, color, 2)
            return
        color_array = np.array(color, dtype=np.float32)
        annotated[binary_mask] = (
            annotated[binary_mask].astype(np.float32) * 0.70 + color_array * 0.30
        ).astype(np.uint8)

    def start_vision(self):
        with self._state_lock:
            if self.vision_enabled:
                self.h["status"]("视觉引擎已在运行。")
                return

            picam2 = None
            try:
                self.h["status"]("正在加载视觉模型...")
                self.ensure_model()

                picam2 = Picamera2()
                config = picam2.create_video_configuration(
                    main={"size": CAMERA_SIZE, "format": "BGR888"},
                    controls={"FrameRate": CAMERA_FPS},
                    queue=False,
                )
                picam2.configure(config)
                picam2.start()

                self.picam2 = picam2
                self.running = True
                self.vision_enabled = True
                self.latest_jpeg = None
                self.pending_status = None
                self.last_frame_captured_at = 0.0
                self._run_token += 1
                run_token = self._run_token
                self.h["vision_state"](True)

                threading.Thread(target=self._capture_thread, args=(run_token,), daemon=True).start()
                threading.Thread(target=self._display_forward_thread, args=(run_token,), daemon=True).start()
                threading.Thread(target=self._display_encode_thread, args=(run_token,), daemon=True).start()
                threading.Thread(target=self._auto_detect_thread, args=(run_token,), daemon=True).start()

                self.h["status"]("摄像头预览已开启，默认低负载预览。")
            except Exception as e:
                self.running = False
                self.vision_enabled = False
                self.picam2 = None
                if picam2 is not None:
                    try:
                        picam2.stop()
                    except Exception:
                        pass
                    try:
                        picam2.close()
                    except Exception:
                        pass
                self.h["vision_state"](False)
                self.h["status"](f"视觉启动失败：{e}")

    def stop_vision(self):
        with self._state_lock:
            self.running = False
            self.vision_enabled = False
            self.detection_paused = False
            self._run_token += 1
            picam2 = self.picam2
            self.picam2 = None

        time.sleep(0.35)

        if picam2:
            for _ in range(2):
                try:
                    picam2.stop()
                except Exception:
                    pass
                try:
                    picam2.close()
                except Exception:
                    pass
                time.sleep(0.1)

        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except Empty:
                break

        while not self.display_queue.empty():
            try:
                self.display_queue.get_nowait()
            except Empty:
                break

        with self._frame_lock:
            self.last_raw_frame = None
            self.last_frame_captured_at = 0.0

        self.latest_jpeg = None
        self.pending_status = None
        time.sleep(0.1)
        gc.collect()
        if "video_clear" in self.h:
            self.h["video_clear"]()
        self.h["vision_state"](False)
        self.h["status"]("视觉引擎已停止。")

    def reset_vision(self):
        self.h["status"]("正在强制重置视觉引擎...")
        self.stop_vision()
        time.sleep(0.6)
        self.start_vision()

    def get_latest_frame_copy(self):
        with self._frame_lock:
            if self.last_raw_frame is None:
                return None, 0.0
            return self.last_raw_frame.copy(), self.last_frame_captured_at

    def analyze_frame(self, frame):
        annotated = frame.copy()
        detected = False
        best_conf = 0.0
        head_lengths = []
        body_lengths = []
        segment_lengths = []

        results = self._run_model(frame)
        detections = self._collect_yolo_detections(frame, results)
        measure_lengths_px = [
            item["pixel_length"]
            for item in detections
            if item["cls_id"] == YOLO_MEASURE_CLASS_ID and item["pixel_length"] > 0
        ]
        mm_per_pixel, used_reference, measure_pixel_length = self._resolve_dynamic_scale(measure_lengths_px)

        for item in detections:
            cls_id = item["cls_id"]
            x1, y1, x2, y2 = item["box"]
            box_conf = item["conf"]
            color = (0, 170, 255) if cls_id == YOLO_HEAD_CLASS_ID else (34, 197, 94)

            if cls_id == YOLO_MEASURE_CLASS_ID:
                measure_color = (255, 0, 0)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), measure_color, 2)
                cv2.putText(
                    annotated,
                    "Measure",
                    (x1, max(y1 - 10, 24)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    measure_color,
                    2,
                )
                continue

            binary_mask = item["binary_mask"]
            polygon = item["polygon"]
            if binary_mask is not None:
                self._apply_mask_overlay(annotated, binary_mask, polygon, color)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 1)
            length_cm = (item["pixel_length"] * mm_per_pixel) / 10.0
            label = "Head" if cls_id == YOLO_HEAD_CLASS_ID else "BL"
            cv2.putText(
                annotated,
                f"{label} {length_cm:.1f} cm",
                (x1, max(y1 - 10, 24)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                color,
                2,
            )
            detected = True
            best_conf = max(best_conf, box_conf)
            segment_lengths.append(length_cm)
            if cls_id == YOLO_HEAD_CLASS_ID:
                head_lengths.append(length_cm)
            else:
                body_lengths.append(length_cm)

        head_length_cm = max(head_lengths) if head_lengths else 0.0
        body_length_cm = max(body_lengths) if body_lengths else 0.0
        total_length_cm = head_length_cm + body_length_cm
        if total_length_cm <= 0 and segment_lengths:
            total_length_cm = max(segment_lengths)

        if total_length_cm > 0:
            cv2.putText(
                annotated,
                f"Head {head_length_cm:.1f} cm | BL {body_length_cm:.1f} cm | Total {total_length_cm:.1f} cm",
                (14, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (255, 255, 255),
                2,
            )

        analysis_meta = {
            "best_conf": best_conf,
            "used_reference": used_reference,
            "head_length_cm": head_length_cm,
            "body_length_cm": body_length_cm,
            "total_length_cm": total_length_cm,
            "mm_per_pixel": mm_per_pixel,
            "measure_pixel_length": measure_pixel_length,
        }
        is_valid = detected and head_length_cm > 0 and body_length_cm > 0
        return is_valid, total_length_cm, annotated, max(10.0 / max(mm_per_pixel, 1e-6), 1e-6), analysis_meta

    def _collect_yolo_detections(self, frame, results):
        detections = []
        valid_classes = {YOLO_HEAD_CLASS_ID, YOLO_BODY_CLASS_ID, YOLO_MEASURE_CLASS_ID}

        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue
            mask_data = None
            polygons = []
            if result.masks is not None and result.masks.data is not None:
                mask_data = result.masks.data.cpu().numpy()
                polygons = result.masks.xy if result.masks.xy is not None else []
            classes = result.boxes.cls.cpu().numpy().astype(int)
            confidences = result.boxes.conf.cpu().numpy()
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()

            for index, cls_id in enumerate(classes):
                if cls_id not in valid_classes:
                    continue
                box_conf = float(confidences[index])
                if box_conf < DETECTION_CONF:
                    continue

                x1, y1, x2, y2 = map(int, boxes_xyxy[index])
                binary_mask = None
                points = None
                polygon = None

                if mask_data is not None and index < len(mask_data):
                    binary_mask, points = self._mask_points_from_array(mask_data[index], frame.shape)
                    if index < len(polygons):
                        polygon_points = np.round(polygons[index]).astype(np.int32)
                        if len(polygon_points) >= 3:
                            polygon = polygon_points

                if points is None and cls_id != YOLO_MEASURE_CLASS_ID:
                    continue

                if points is not None:
                    rect = cv2.minAreaRect(points)
                    pixel_length = float(max(rect[1]))
                else:
                    pixel_length = float(max(x2 - x1, y2 - y1))

                if pixel_length <= 0:
                    continue

                detections.append(
                    {
                        "cls_id": cls_id,
                        "conf": box_conf,
                        "box": (x1, y1, x2, y2),
                        "binary_mask": binary_mask,
                        "polygon": polygon,
                        "pixel_length": pixel_length,
                    }
                )

        return detections

    def _resolve_dynamic_scale(self, measure_lengths_px):
        mm_per_pixel = self.mm_per_pixel if self.mm_per_pixel > 0 else DEFAULT_MM_PER_PIXEL
        used_reference = False
        measure_pixel_length = self._robust_average(measure_lengths_px)

        if measure_pixel_length > 0:
            mm_per_pixel = MEASURE_REAL_LENGTH_MM / max(measure_pixel_length, 1e-6)
            used_reference = True

        if mm_per_pixel <= 0:
            mm_per_pixel = DEFAULT_MM_PER_PIXEL

        self.mm_per_pixel = mm_per_pixel
        self.pixels_per_cm = 10.0 / mm_per_pixel
        return mm_per_pixel, used_reference, measure_pixel_length

    def collect_burst_frames(self, count, interval):
        frames = []
        last_seen = 0.0

        for index in range(count):
            deadline = time.time() + max(interval * 2.5, 0.4)
            captured = None

            while time.time() < deadline:
                frame, frame_time = self.get_latest_frame_copy()
                if frame is not None and frame_time > last_seen:
                    captured = frame
                    last_seen = frame_time
                    break
                time.sleep(0.01)

            if captured is None:
                frame, _ = self.get_latest_frame_copy()
                if frame is not None:
                    captured = frame

            if captured is not None:
                frames.append(captured)

            if index < count - 1:
                time.sleep(interval)

        return frames

    def analyze_burst_frames(self, frames):
        valid_results = []

        for index, frame in enumerate(frames, start=1):
            detected, length_cm, annotated, _, meta = self.analyze_frame(frame)
            item = {
                "index": index,
                "detected": detected,
                "length_cm": length_cm,
                "annotated": annotated,
                "best_conf": meta["best_conf"],
                "used_reference": meta["used_reference"],
                "head_length_cm": meta["head_length_cm"],
                "body_length_cm": meta["body_length_cm"],
            }
            if detected:
                valid_results.append(item)

        if not valid_results:
            return {
                "detected": False,
                "final_length": 0.0,
                "best_result": None,
                "valid_count": 0,
                "total_count": len(frames),
                "final_head_length": 0.0,
                "final_body_length": 0.0,
            }

        final_head_length = self._robust_average([item["head_length_cm"] for item in valid_results])
        final_body_length = self._robust_average([item["body_length_cm"] for item in valid_results])
        final_length = final_head_length + final_body_length
        if final_length <= 0:
            final_length = self._robust_average([item["length_cm"] for item in valid_results])

        best_result = sorted(
            valid_results,
            key=lambda item: (
                0 if item["used_reference"] else 1,
                abs(item["length_cm"] - final_length),
                -item["best_conf"],
            ),
        )[0]

        return {
            "detected": True,
            "final_length": final_length,
            "best_result": best_result,
            "valid_count": len(valid_results),
            "total_count": len(frames),
            "final_head_length": final_head_length,
            "final_body_length": final_body_length,
        }

    def _capture_thread(self, run_token):
        while self.running and self.picam2 and run_token == self._run_token:
            try:
                frame = self.picam2.capture_array()
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame = cv2.resize(frame, DISPLAY_SIZE)
                with self._frame_lock:
                    self.last_raw_frame = frame.copy()
                    self.last_frame_captured_at = time.time()

                if self.frame_queue.full():
                    self.frame_queue.get_nowait()
                self.frame_queue.put_nowait(frame)
            except Exception as e:
                if self.running:
                    self.h.get("status", lambda x: None)(f"摄像头捕帧异常：{e}")
                time.sleep(0.01)

    def _display_forward_thread(self, run_token):
        last_overlay_at = 0.0
        while self.running and run_token == self._run_token:
            try:
                frame = self.frame_queue.get(timeout=1)
            except Empty:
                continue

            output_frame = frame
            overlay_enabled_getter = self.h.get("live_overlay_enabled")
            overlay_enabled = overlay_enabled_getter() if overlay_enabled_getter else False
            if overlay_enabled:
                interval = 1.0 / LIVE_OVERLAY_MAX_FPS
                now = time.time()
                wait_s = interval - (now - last_overlay_at)
                if wait_s > 0:
                    time.sleep(wait_s)
                last_overlay_at = time.time()
                try:
                    _, _, output_frame, _, _ = self.analyze_frame(frame)
                except Exception as e:
                    output_frame = frame
                    self.pending_status = f"实时渲染异常：{e}"
            else:
                last_overlay_at = 0.0

            if self.display_queue.full():
                self.display_queue.get_nowait()
            self.display_queue.put_nowait(output_frame)

    def _display_encode_thread(self, run_token):
        while self.running and run_token == self._run_token:
            try:
                frame = self.display_queue.get(timeout=0.1)
            except Empty:
                continue

            try:
                success, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
                if success:
                    self.latest_jpeg = jpeg.tobytes()
            except Exception as e:
                if self.running:
                    self.h.get("status", lambda x: None)(f"图像编码异常：{e}")

    def _auto_detect_thread(self, run_token):
        last_detect_at = 0.0
        while self.running and run_token == self._run_token:
            time.sleep(0.02)

            if self.detection_paused:
                continue

            auto_enabled_getter = self.h.get("auto_enabled")
            detect_callback = self.h.get("detection_event")
            if not auto_enabled_getter or not detect_callback:
                continue
            if not auto_enabled_getter():
                continue

            now = time.time()
            if (now - last_detect_at) < AUTO_DETECT_INTERVAL:
                continue

            if self._snapshot_lock.locked():
                continue

            acquired = self._auto_detect_lock.acquire(blocking=False)
            if not acquired:
                continue

            try:
                last_detect_at = time.time()
                frames = self.collect_burst_frames(AUTO_DETECT_BURST_COUNT, AUTO_DETECT_BURST_INTERVAL)
                if not frames:
                    detect_callback(False, 0.0)
                    continue

                result = self.analyze_burst_frames(frames)
                detect_callback(result["detected"], result["final_length"])
            except Exception as e:
                self.pending_status = f"自动检测异常：{e}"
            finally:
                self._auto_detect_lock.release()


# =========================================================
# Main UI / workflow
# =========================================================


# =========================================================
# 操作权互斥锁
# =========================================================
class OperationLock:
    TIMEOUT = 30.0   # 空闲超时自动释放（秒）

    def __init__(self):
        self._lock = threading.Lock()
        self._holder: str = ""          # "PC" / "手机" / ""
        self._acquired_at: float = 0.0
        self._persistent: bool = False

    def acquire(self, side: str, persistent: bool = False) -> bool:
        with self._lock:
            now = time.time()
            # 已超时 → 自动释放
            if self._holder and not self._persistent and (now - self._acquired_at) > self.TIMEOUT:
                self._holder = ""
                self._acquired_at = 0.0
                self._persistent = False
            # 空闲 or 同一端续期
            if not self._holder or self._holder == side:
                self._holder = side
                self._acquired_at = now
                self._persistent = self._persistent or persistent
                return True
            return False

    def release(self):
        with self._lock:
            self._holder = ""
            self._acquired_at = 0.0
            self._persistent = False

    def holder(self) -> str:
        with self._lock:
            now = time.time()
            if self._holder and not self._persistent and (now - self._acquired_at) > self.TIMEOUT:
                self._holder = ""
                self._acquired_at = 0.0
                self._persistent = False
            return self._holder

    def is_free(self) -> bool:
        return self.holder() == ""

    def is_held_by(self, side: str) -> bool:
        return self.holder() == side

@dataclass
class SystemState:
    auto_enabled: bool = False
    sequence_running: bool = False
    snapshot_running: bool = False
    job_running: bool = False
    job_task_sent: bool = False
    stable_count: int = 0
    last_stable_length: Optional[float] = None
    last_trigger_time: float = 0.0
    absent_count: int = 0
    target_present_latch: bool = False

    current_pos_mm: float = 0.0
    serial_connected: bool = False
    serial_port: str = "未连接"
    vision_running: bool = False
    live_overlay_enabled: bool = False
    general_status: str = "系统初始化中..."
    live_result_text: str = "--"
    live_detail_text: str = "等待视频流"
    auto_state_text: str = "自动流程待机"
    last_detected_length: float = 0.0
    last_detection_at: Optional[float] = None
    last_snapshot_path: str = ""
    last_snapshot_length: float = 0.0
    last_snapshot_head_length: float = 0.0
    last_snapshot_body_length: float = 0.0
    last_snapshot_time: Optional[float] = None
    last_snapshot_jpeg: Optional[bytes] = None
    current_weight_g: float = 0.0
    last_snapshot_weight_g: float = 0.0
    last_weight_at: Optional[float] = None
    weight_poll_enabled: bool = True
    weight_poll_interval: float = 0.8
    weight_poll_running: bool = True
    limit_home_state: Optional[bool] = None
    limit_stop_state: Optional[bool] = None
    cut_start_state: Optional[bool] = None
    cut_end_state: Optional[bool] = None
    arduino_ready: bool = False
    job_state_text: str = "待机"
    job_progress_text: str = "--"
    job_result_text: str = "最近任务：--"
    last_serial_line: str = "--"
    cpu_percent: float = 0.0
    mem_percent: float = 0.0
    cpu_temp: float = 0.0
    all_motor_speed: int = 0
    motor1_speed: int = 0
    motor2_speed: int = 0
    servo_angle_deg: int = DEFAULT_SERVO_CENTER
    servo2_angle_deg: int = DEFAULT_SERVO2_UP

    current_total_mm: float = 0.0
    current_head_mm: float = 0.0
    current_remain_mm: float = 0.0
    generated_tasks: List[Dict[str, Any]] = field(default_factory=list)
    auto_job_mode: str = "fixed"

    measurement_rows: List[Dict[str, str]] = field(default_factory=list)
    last_history_record_at: float = 0.0
    last_history_length: Optional[float] = None
    pending_logs: List[str] = field(default_factory=list)
    disconnect_stop_requested: bool = False
    disconnect_requested_at: Optional[float] = None
    disconnect_grace_seconds: float = DISCONNECT_PROTECT_GRACE_SECONDS
    serial_reconnect_running: bool = True
    serial_reconnect_interval: float = 2.0
    estop_weight_suppress_until: float = 0.0
    record_count: int = 0
    length_sum: float = 0.0
    max_detected_length: float = 0.0
    auto_runs_count: int = 0
    snapshots_taken: int = 0
    session_started_at: float = field(default_factory=time.time)

    cut_mode: str = "fixed"
    voice_broadcast: bool = DEFAULT_VOICE_BROADCAST
    fixed_length_mm: int = 60
    avg_parts_count: int = 4
    auto_speed: int = DEFAULT_FEED_SPEED
    stable_frames: int = DEFAULT_STABLE_FRAMES
    absent_frames: int = DEFAULT_ABSENT_FRAMES
    tolerance: float = DEFAULT_TOLERANCE
    cooldown: float = DEFAULT_COOLDOWN
    settle_ms: int = DEFAULT_SETTLE_MS
    sort_hold_ms: int = DEFAULT_SORT_HOLD_MS
    center_hold_ms: int = DEFAULT_CENTER_HOLD_MS
    eject_ms: int = DEFAULT_EJECT_MS
    cut_time_ms: int = DEFAULT_CUT_TIME_MS
    servo_center: int = DEFAULT_SERVO_CENTER
    servo_head: int = DEFAULT_SERVO_HEAD
    servo_body: int = DEFAULT_SERVO_BODY
    servo2_down: int = DEFAULT_SERVO2_DOWN
    servo2_up: int = DEFAULT_SERVO2_UP
    cut_motor_speed: int = DEFAULT_CUT_MOTOR_SPEED
    pump_cut_speed: int = DEFAULT_PUMP_CUT_SPEED
    blade_offset_mm: float = DEFAULT_BLADE_OFFSET_MM
    step_dist_mm: float = 10.0

    weight_poll_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    serial_reconnect_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    speech_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    op_lock: OperationLock = field(default_factory=OperationLock, repr=False)


class AppController:
    def __init__(self, state: SystemState):
        self.state = state
        self.serial: Optional[SerialManager] = None
        self.vision: Optional[VisionController] = None
        self._config_lock = threading.Lock()
        self._log_file_lock = threading.Lock()
        self._load_persistent_logs()
        self._load_persisted_config()
        self.init_subsystems()

    def init_subsystems(self):
        self.serial = SerialManager(
            pos_callback=self.update_pos,
            log_callback=self.add_log,
            state_callback=self.on_serial_state_change,
            line_callback=self.on_serial_line,
        )
        self.vision = VisionController(
            {
                "video_clear": self._safe_video_clear,
                "status": self.on_system_status,
                "vision_state": self.on_vision_state_change,
                "auto_enabled": lambda: self.state.auto_enabled,
                "live_overlay_enabled": lambda: self.state.live_overlay_enabled,
                "detection_event": self.on_detection_event,
            }
        )
        threading.Thread(target=self._weight_poll_loop, daemon=True).start()
        threading.Thread(target=self._serial_reconnect_loop, daemon=True).start()
        threading.Thread(target=self._vision_status_loop, daemon=True).start()
        threading.Thread(target=self._hardware_monitor_loop, daemon=True).start()

    def _safe_video_clear(self):
        return

    def add_log(self, msg: str):
        timestamp = time.strftime("%H:%M:%S")
        entry = f"[{timestamp}] {msg}"
        self.state.pending_logs.append(entry)
        if len(self.state.pending_logs) > 60:
            self.state.pending_logs.pop(0)
        self._append_log_file(entry)

    def _format_side_log(self, side: str, msg: str) -> str:
        return f"[{side}] {msg}"

    def _append_log_file(self, entry: str):
        try:
            with self._log_file_lock:
                with open(RUN_LOG_FILE_PATH, "a", encoding="utf-8") as f:
                    f.write(entry + "\n")
        except Exception:
            pass

    def _load_persistent_logs(self):
        if not os.path.exists(RUN_LOG_FILE_PATH):
            return
        try:
            with open(RUN_LOG_FILE_PATH, "r", encoding="utf-8", errors="ignore") as f:
                lines = [line.rstrip("\r\n") for line in f.readlines()[-LOG_HISTORY_LIMIT:]]
            self.state.pending_logs.extend([line for line in lines if line.strip()])
        except Exception:
            pass

    def _get_config_snapshot(self) -> Dict[str, Any]:
        return {field: getattr(self.state, field) for field in PERSISTED_CONFIG_FIELDS}

    def _apply_config_values(
        self,
        payload: Dict[str, Any],
        *,
        persist: bool = False,
        log_message: Optional[str] = None,
    ):
        for key, value in payload.items():
            if not hasattr(self.state, key):
                continue
            try:
                if key in CONFIG_INT_FIELDS:
                    setattr(self.state, key, int(value))
                elif key in CONFIG_FLOAT_FIELDS:
                    setattr(self.state, key, float(value))
                elif key in CONFIG_BOOL_FIELDS:
                    setattr(self.state, key, bool(value))
                elif key in CONFIG_STR_FIELDS:
                    setattr(self.state, key, str(value))
            except Exception:
                continue

        if "auto_enabled" in payload:
            self.on_auto_change(bool(payload.get("auto_enabled")))

        if persist:
            self._save_persisted_config()
        if log_message:
            self.add_log(log_message)

    def _save_persisted_config(self):
        try:
            with self._config_lock:
                with open(CONFIG_FILE_PATH, "w", encoding="utf-8") as f:
                    json.dump(self._get_config_snapshot(), f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.add_log(f"本地参数保存失败：{e}")

    def _load_persisted_config(self):
        if not os.path.exists(CONFIG_FILE_PATH):
            return
        try:
            with open(CONFIG_FILE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                self._apply_config_values(data, persist=False, log_message=None)
                self.add_log("已加载本地参数配置。")
        except Exception as e:
            self.add_log(f"本地参数加载失败：{e}")

    def reset_config(self, side: str = "系统"):
        self._apply_config_values(
            DEFAULT_CONFIG,
            persist=True,
            log_message=self._format_side_log(side, "参数已恢复为默认值。"),
        )

    def start_vision(self, side: str = "系统"):
        if self.vision is not None:
            self.add_log(self._format_side_log(side, "正在启动视觉。"))
            self.vision.start_vision()

    def stop_vision(self, side: str = "系统"):
        if self.vision is not None:
            self.add_log(self._format_side_log(side, "正在停止视觉。"))
            self.vision.stop_vision()

    def set_live_overlay(self, enabled: bool, side: str = "系统"):
        self.state.live_overlay_enabled = bool(enabled)
        if self.state.live_overlay_enabled:
            self.add_log(self._format_side_log(side, "已开启实时渲染。"))
        else:
            self.add_log(self._format_side_log(side, "已关闭实时渲染。"))

    def _build_length_trend_points(self) -> List[Dict[str, Any]]:
        points = []
        rows = list(reversed(self.state.measurement_rows[-8:]))
        for row in rows:
            try:
                value = float(str(row.get("length", "")).replace("cm", "").strip())
            except Exception:
                continue
            points.append(
                {
                    "time": row.get("time", "--"),
                    "length_cm": round(value, 2),
                    "source": row.get("source", "--"),
                }
            )
        return points

    def _number_to_wavs(self, num: float) -> List[str]:
        wavs = []
        int_part = int(num)
        dec_part = int(round((num - int_part) * 10))

        if dec_part >= 10:
            int_part += 1
            dec_part = 0

        if int_part == 0:
            wavs.append("0.wav")
        else:
            s = str(int_part)
            length = len(s)
            for i, char in enumerate(s):
                digit = int(char)
                pos = length - i - 1
                if digit != 0:
                    if pos == 1 and digit == 1 and length == 2:
                        wavs.append("shi.wav")
                    else:
                        wavs.append(f"{digit}.wav")
                        if pos == 1:
                            wavs.append("shi.wav")
                        elif pos == 2:
                            wavs.append("bai.wav")
                        elif pos == 3:
                            wavs.append("qian.wav")
                else:
                    if pos != 0 and i + 1 < length and s[i + 1] != "0":
                        wavs.append("0.wav")

        wavs.append("dian.wav")
        wavs.append(f"{dec_part}.wav")
        return wavs

    def play_voice_sequence(self, wav_files: List[str]):
        if not self.state.voice_broadcast or not wav_files:
            return

        def _play():
            with self.state.speech_lock:
                for wav in wav_files:
                    filepath = os.path.join(VOICE_AUDIO_DIR, wav)
                    try:
                        if not os.path.exists(filepath):
                            self.add_log(f"语音文件不存在：{filepath}")
                            continue
                        subprocess.run(["aplay", "-q", filepath], check=False)
                    except FileNotFoundError:
                        self.add_log("未找到 aplay，无法执行 Linux 语音播报。")
                        break
                    except Exception as e:
                        self.add_log(f"语音播放失败：{wav} -> {e}")

        threading.Thread(target=_play, daemon=True).start()

    def on_system_status(self, msg: str):
        self.state.general_status = msg

    def on_serial_state_change(self, connected: bool, port: Optional[str]):
        self.state.serial_connected = connected
        self.state.serial_port = port or "未连接"
        if not connected:
            self.state.arduino_ready = False

    def on_serial_line(self, line: str):
        self.state.last_serial_line = line
        if line.startswith("UNO R4 READY"):
            self.state.arduino_ready = True
        if line.startswith("LIMITS"):
            try:
                parts = line.replace("LIMITS", "").strip().split()
                for part in parts:
                    if part.startswith("L1="):
                        self.state.limit_home_state = part.split("=")[1] == "1"
                    elif part.startswith("L2="):
                        self.state.limit_stop_state = part.split("=")[1] == "1"
                    elif part.startswith("CUT_START="):
                        self.state.cut_start_state = part.split("=")[1] == "1"
                    elif part.startswith("CUT_END="):
                        self.state.cut_end_state = part.split("=")[1] == "1"
            except Exception:
                pass
        if line.startswith("WEIGHT="):
            weight_g = self._parse_weight_line(line)
            if weight_g is not None:
                self.state.current_weight_g = weight_g
                self.state.last_weight_at = time.time()
        if line.startswith("JOBSTATE"):
            self.state.job_state_text = line.replace("JOBSTATE", "").strip()
            self.state.job_running = True
            if self.state.generated_tasks:
                self.state.job_progress_text = f"任务总刀数：{len(self.state.generated_tasks)}"
        elif line.startswith("JOBDONE"):
            self.state.job_running = False
            self.state.sequence_running = False
            self.state.target_present_latch = False
            self.state.all_motor_speed = 0
            self.state.motor1_speed = 0
            self.state.motor2_speed = 0
            self.state.job_state_text = "任务完成"
            self.state.job_result_text = "最近任务：处理完成"
            self.state.auto_state_text = "自动流程完成，等待下一条鱼。"
            self.state.auto_runs_count += 1
        elif line.startswith("JOBERROR"):
            self.state.job_running = False
            self.state.sequence_running = False
            self.state.target_present_latch = False
            self.state.all_motor_speed = 0
            self.state.motor1_speed = 0
            self.state.motor2_speed = 0
            self.state.job_state_text = line.replace("JOBERROR", "").strip()
            self.state.job_result_text = f"最近任务：异常 {self.state.job_state_text}"
            self.state.auto_state_text = f"自动流程异常：{self.state.job_state_text}"
        elif line.startswith("STOPPED"):
            self.state.job_running = False
            self.state.sequence_running = False
            self.state.all_motor_speed = 0
            self.state.motor1_speed = 0
            self.state.motor2_speed = 0
            self.state.job_state_text = "已停止"
            self.state.job_result_text = "最近任务：人工停止"
        elif line.startswith("TARE_OK"):
            self.state.current_weight_g = 0.0
            self.state.last_weight_at = time.time()
            self.add_log("称重模块去皮完成。")
        elif line == "ZERO SET":
            self.state.current_pos_mm = 0.0
        elif line.startswith("CUT_OK"):
            self.state.job_state_text = "横切完成"
            self.state.job_result_text = "最近切割：横切完成"
        elif line.startswith("CUTHOME_OK"):
            self.state.job_state_text = "刀头已复位"
            self.state.job_result_text = "最近切割：刀头已回到起点"
        elif line.startswith("CUT_FORWARD_OK"):
            self.state.job_state_text = "刀头已推进到终点"
            self.state.job_result_text = "最近切割：刀头推进完成"
        elif line.startswith("CUT_REVERSE_OK"):
            self.state.job_state_text = "刀头已反向回起点"
            self.state.job_result_text = "最近切割：刀头回退完成"
        elif line.startswith("CUT_ERR"):
            self.state.job_running = False
            self.state.sequence_running = False
            err = line.replace("CUT_ERR", "").strip()
            self.state.job_state_text = err or "横切异常"
            self.state.job_result_text = f"最近切割：异常 {self.state.job_state_text}"
            self.state.auto_state_text = f"切割机构异常：{self.state.job_state_text}"
        elif line.startswith("JOBSTATUS"):
            try:
                parts = line.replace("JOBSTATUS", "").strip().split()
                for part in parts:
                    if part.startswith("cutStart="):
                        self.state.cut_start_state = part.split("=")[1] == "1"
                    elif part.startswith("cutEnd="):
                        self.state.cut_end_state = part.split("=")[1] == "1"
            except Exception:
                pass

        if (
            line.startswith("JOBDONE")
            or line.startswith("JOBERROR")
            or line.startswith("STOPPED")
            or line.startswith("DONE STEP POS=")
            or line.startswith("HOME_OK")
            or line == "ZERO SET"
            or line.startswith("OK FEEDSTOP")
            or line.startswith("EJECT_OK")
            or line.startswith("OK ALL")
            or line.startswith("OK M1")
            or line.startswith("OK M2")
            or line.startswith("OK FLIP")
            or line.startswith("OK FLIP2")
            or line.startswith("OK SAW")
            or line.startswith("OK PUMP")
            or line.startswith("SERVO_OK")
            or line.startswith("CUT_OK")
            or line.startswith("CUTHOME_OK")
            or line.startswith("CUT_FORWARD_OK")
            or line.startswith("CUT_REVERSE_OK")
            or line.startswith("CUT_ERR")
            or line.startswith("ERR HOME_TIMEOUT")
            or line.startswith("ERR SOFT_LIMIT_EXCEEDED")
            or line.startswith("ERR LIMIT_HOME_HIT")
            or line.startswith("ERR LIMIT_STOP_HIT")
            or line.startswith("ERR BUSY")
            or line.startswith("ERR RECURSION_BLOCKED")
            or line.startswith("ERR UNKNOWN CMD")
            or line.startswith("ERR CFG")
        ):
            self.state.op_lock.release()

    def on_vision_state_change(self, running: bool):
        self.state.vision_running = running
        if not running:
            self.state.live_result_text = "--"
            self.state.live_detail_text = "等待视频流"
        else:
            self.state.live_result_text = "预览中"
            self.state.live_detail_text = "摄像头已开启，默认只预览；按“连拍 5 张测长”才识别。打开自动开关后，才启用后台低频自动检测。"

    def update_pos(self, pos_mm: float):
        self.state.current_pos_mm = pos_mm

    def _vision_status_loop(self):
        while True:
            time.sleep(0.15)
            if self.vision is None:
                continue
            pending_status = getattr(self.vision, "pending_status", None)
            if pending_status:
                self.state.general_status = pending_status
                self.vision.pending_status = None

    def _hardware_monitor_loop(self):
        while True:
            time.sleep(2.0)
            try:
                self.state.cpu_percent = float(psutil.cpu_percent())
            except Exception:
                self.state.cpu_percent = 0.0
            try:
                self.state.mem_percent = float(psutil.virtual_memory().percent)
            except Exception:
                self.state.mem_percent = 0.0
            try:
                with open("/sys/class/thermal/thermal_zone0/temp", "r", encoding="utf-8") as f:
                    self.state.cpu_temp = float(f.read().strip()) / 1000.0
            except Exception:
                self.state.cpu_temp = 0.0

    def _serial_reconnect_loop(self):
        while self.state.serial_reconnect_running:
            time.sleep(self.state.serial_reconnect_interval)
            if self.serial is None or self.state.serial_connected:
                continue
            if self.state.serial_reconnect_lock.locked():
                continue
            acquired = self.state.serial_reconnect_lock.acquire(blocking=False)
            if not acquired:
                continue
            try:
                recovered = self.serial.reconnect(silent=True)
                if recovered:
                    self.add_log(f"Arduino reconnected automatically: {self.serial.port}")
                    self.state.arduino_ready = False
                    self.serial.send_cmd("LIMITS")
                    self.serial.send_cmd("JOBSTATUS")
            except Exception as e:
                self.add_log(f"Serial auto-reconnect error: {e}")
            finally:
                self.state.serial_reconnect_lock.release()

    def _parse_weight_line(self, line: str) -> Optional[float]:
        if not line.startswith("WEIGHT="):
            return None
        try:
            return float(line.split("=", 1)[1].strip().split()[0])
        except Exception:
            return None

    def _format_weight_text(self, weight_g: Optional[float]) -> str:
        if weight_g is None or weight_g <= 0:
            return "--"
        if weight_g >= 1000:
            return f"{weight_g / 1000.0:.2f} kg"
        return f"{weight_g:.1f} g"

    def query_weight_g(self, log_on_fail: bool = True, retries: int = 3) -> Optional[float]:
        if self.serial is None:
            return None
        last_error = "未收到称重数据"
        for attempt in range(max(1, retries)):
            line = self.serial.send_cmd_get_line(
                "WEIGHT",
                wait_prefixes=("WEIGHT=", "WEIGHT_ERR"),
                reject_prefixes=("ERR ", "JOBERROR", "STOPPED"),
                timeout=2.0,
                silent=not log_on_fail,
            )
            if not line:
                last_error = "未收到称重数据"
            elif line.startswith("WEIGHT_ERR"):
                detail = line.replace("WEIGHT_ERR", "").strip()
                last_error = f"HX711 超时{f'：{detail}' if detail else ''}"
            else:
                weight_g = self._parse_weight_line(line)
                if weight_g is not None:
                    return weight_g
                last_error = f"返回格式异常 -> {line}"
            if attempt < retries - 1:
                time.sleep(0.12)
        if log_on_fail:
            self.add_log(f"重量读取失败：{last_error}。")
        return None

    def _weight_poll_loop(self):
        while self.state.weight_poll_running:
            busy = self.state.snapshot_running or self.state.sequence_running or self.state.job_running
            time.sleep(4.0 if busy else self.state.weight_poll_interval)
            if not self.state.weight_poll_enabled or self.serial is None or not self.state.serial_connected:
                continue
            if time.time() < self.state.estop_weight_suppress_until:
                continue
            if self.state.weight_poll_lock.locked():
                continue
            acquired = self.state.weight_poll_lock.acquire(blocking=False)
            if not acquired:
                continue
            try:
                self.serial._suppress_weight_log_count = max(self.serial._suppress_weight_log_count, 6)
                weight_g = self.query_weight_g(log_on_fail=False, retries=2)
                if weight_g is not None:
                    self.state.current_weight_g = weight_g
                    self.state.last_weight_at = time.time()
            except Exception as e:
                self.add_log(f"实时称重异常：{e}")
                time.sleep(1.0)
            finally:
                self.state.weight_poll_lock.release()
    def on_auto_change(self, enabled: bool):
        self.state.auto_enabled = bool(enabled)
        self.state.stable_count = 0
        self.state.last_stable_length = None
        self.state.absent_count = 0
        self.state.target_present_latch = False
        if self.state.auto_enabled:
            self.state.auto_state_text = "视觉自动加工已开启，等待目标稳定。"
            self.add_log("自动模式已开启：后台低频检测启动，稳定后会自动测长并下发加工。")
        else:
            self.state.auto_state_text = "自动加工已关闭。"
            self.add_log("自动模式已关闭，恢复为默认手动测长模式。")

    def clear_measurements(self, side: str = "系统"):
        self.state.measurement_rows = []
        self.add_log(self._format_side_log(side, "已清空最近识别记录。"))

    def clear_generated_tasks(self, side: str = "系统"):
        self.state.generated_tasks = []
        self.state.current_total_mm = 0.0
        self.state.current_head_mm = 0.0
        self.state.current_remain_mm = 0.0
        self.add_log(self._format_side_log(side, "任务表已清空。"))

    def update_config(self, payload: Dict[str, Any], side: str = "系统"):
        self._apply_config_values(
            payload,
            persist=True,
            log_message=self._format_side_log(side, "参数已更新。"),
        )

    def _manual_action_locked(self, action_name: str, side: str = "系统") -> bool:
        if self.state.sequence_running or self.state.job_running:
            self.add_log(self._format_side_log(side, f"任务执行中，禁止手动{action_name}；如需干预请先急停。"))
            return True
        if not self.state.op_lock.acquire(side):
            holder = self.state.op_lock.holder()
            self.add_log(self._format_side_log(side, f"操作权被【{holder}端】持有，{action_name}已拦截。请等待对方操作完成（30s 自动释放）。"))
            return True
        return False

    def _validate_generated_tasks(self, tasks: List[Dict[str, Any]]):
        if not tasks:
            return False, "没有可下发的切割任务。"
        if len(tasks) > ARDUINO_MAX_TASKS:
            return False, f"任务数 {len(tasks)} 超出 Arduino 上限 {ARDUINO_MAX_TASKS}。"
        for task in tasks:
            length_mm = float(task["len_mm"])
            if length_mm <= 0:
                return False, f"存在无效任务长度：{task['len_mm']}"
            if length_mm > STEPPER_SOFT_LIMIT_MM:
                return False, f"单刀长度 {length_mm:.2f} mm 超出丝杠安全行程 {STEPPER_SOFT_LIMIT_MM:.2f} mm。"
        return True, ""

    def _send_command_checked(self, cmd: str, wait_prefixes, label: str, timeout: float = 2.5, side: str = "系统"):
        if self.serial is None:
            return False
        ok = self.serial.send_cmd(cmd, wait_prefixes=wait_prefixes, timeout=timeout)
        if not ok:
            self.add_log(self._format_side_log(side, f"{label}失败：{cmd}"))
        return ok

    def _send_locked_manual_command(self, cmd: str, action_name: str, success_log: str, side: str = "系统") -> bool:
        if self.serial is None:
            self.add_log(self._format_side_log(side, f"{action_name}失败：Arduino 未连接。"))
            self.state.op_lock.release()
            return False
        ok = self.serial.send_cmd(cmd)
        if not ok:
            self.add_log(self._format_side_log(side, f"{action_name}失败：串口命令未发送成功。"))
            self.state.op_lock.release()
            return False
        if success_log:
            self.add_log(self._format_side_log(side, success_log))
        return True

    def generate_tasks_from_latest(self, side: str = "系统"):
        if self.state.last_snapshot_length <= 0:
            self.add_log(self._format_side_log(side, "请先完成一次测长，再生成刀长任务表。"))
            self.clear_generated_tasks(side=side)
            return
        head_mm = round(self.state.last_snapshot_head_length * 10.0, 2)
        remain_mm = round(self.state.last_snapshot_body_length * 10.0, 2)
        total_mm = round(head_mm + remain_mm, 2)
        if head_mm <= 0 or remain_mm <= 0 or total_mm <= 0:
            self.add_log(self._format_side_log(side, "生成任务失败：当前分割结果未同时得到有效鱼头和鱼身长度。"))
            self.clear_generated_tasks(side=side)
            return
        tasks = [{"idx": 1, "kind": "鱼头", "len_mm": round(head_mm, 2), "tag": "HEAD"}]
        if self.state.cut_mode == "avg":
            parts = int(self.state.avg_parts_count or 0)
            if parts <= 0:
                self.add_log(self._format_side_log(side, "生成任务失败：平均分段数必须大于 0。"))
                self.clear_generated_tasks(side=side)
                return
            if remain_mm > 0:
                each = round(remain_mm / parts, 2)
                for _ in range(parts):
                    tasks.append({"idx": len(tasks) + 1, "kind": "鱼段", "len_mm": each, "tag": "BODY"})
            remainder_mm = 0.0
        else:
            seg_len = float(self.state.fixed_length_mm or 0)
            if seg_len <= 0:
                self.add_log(self._format_side_log(side, "生成任务失败：固定分段长度必须大于 0。"))
                self.clear_generated_tasks(side=side)
                return
            if remain_mm > 0:
                count = int(remain_mm // seg_len)
                for _ in range(count):
                    tasks.append({"idx": len(tasks) + 1, "kind": "鱼段", "len_mm": round(seg_len, 2), "tag": "BODY"})
                remainder_mm = round(remain_mm - count * seg_len, 2)
            else:
                remainder_mm = 0.0
        valid, error_text = self._validate_generated_tasks(tasks)
        if not valid:
            self.add_log(self._format_side_log(side, f"生成任务失败：{error_text}"))
            self.clear_generated_tasks(side=side)
            return
        self.state.current_total_mm = total_mm
        self.state.current_head_mm = head_mm
        self.state.current_remain_mm = remain_mm
        self.state.generated_tasks = tasks
        summary = f"总长 {total_mm:.2f} mm，鱼头 {head_mm:.2f} mm，剩余 {remain_mm:.2f} mm。生成 {len(tasks)} 刀，尾料忽略 {remainder_mm:.2f} mm。"
        self.add_log(self._format_side_log(side, f"已生成刀长任务表：{summary}"))

    def dispatch_job_only(self, side: str = "系统"):
        return self._dispatch_job_v2(start_now=False, side=side)

    def dispatch_and_start_job(self, side: str = "系统"):
        return self._dispatch_job_v2(start_now=True, side=side)

    def _dispatch_job_v2(self, start_now: bool, side: str = "系统"):
        if not self.state.serial_connected or self.serial is None:
            self.add_log(self._format_side_log(side, "下发任务失败：Arduino 未连接。"))
            return False
        if self.state.job_running:
            self.add_log(self._format_side_log(side, "下发任务失败：当前加工任务仍在执行中，请先等待完成或急停。"))
            return False
        if not self.state.generated_tasks:
            self.add_log(self._format_side_log(side, "下发任务失败：请先生成刀长任务表。"))
            return False
        if not self.state.op_lock.acquire(side, persistent=start_now):
            holder = self.state.op_lock.holder()
            self.add_log(self._format_side_log(side, f"操作权被【{holder}端】持有，任务下发已拦截。"))
            return False
        keep_lock_for_running_job = False
        try:
            valid, error_text = self._validate_generated_tasks(self.state.generated_tasks)
            if not valid:
                self.add_log(self._format_side_log(side, f"下发任务失败：{error_text}"))
                return False
            ok = self._send_command_checked("JOBCLEAR", ("JOBCLEARED",), "清空任务", timeout=2.0, side=side)
            if not ok:
                self.state.job_task_sent = False
                self.state.job_running = False
                self.state.sequence_running = False
                self.state.auto_state_text = "任务下发失败，请检查日志后重试。"
                return False
            for task in self.state.generated_tasks:
                ok = self._send_command_checked(
                    f"JOBADD {task['len_mm']:.2f} {task['tag']}",
                    ("JOBADD_OK",),
                    f"下发第 {task['idx']} 刀",
                    timeout=2.0,
                    side=side,
                )
                if not ok:
                    self.serial.send_cmd("JOBCLEAR", wait_prefixes=("JOBCLEARED",), timeout=1.5)
                    self.state.job_task_sent = False
                    self.state.job_running = False
                    self.state.sequence_running = False
                    self.state.auto_state_text = "任务下发失败，请检查日志后重试。"
                    return False
            cfg_items = [
                (f"CFG FEED {int(self.state.auto_speed or DEFAULT_FEED_SPEED)}", "设置送料速度"),
                (f"CFG EJECT {int(self.state.eject_ms or DEFAULT_EJECT_MS)}", "设置尾料排空时间"),
                (f"CFG SETTLE {int(self.state.settle_ms or DEFAULT_SETTLE_MS)}", "设置挡板稳定时间"),
                (f"CFG SORTHOLD {int(self.state.sort_hold_ms or DEFAULT_SORT_HOLD_MS)}", "设置分拣停留时间"),
                (f"CFG CENTERHOLD {int(self.state.center_hold_ms or DEFAULT_CENTER_HOLD_MS)}", "设置回中位稳定时间"),
                (f"CFG CUTSPEED {int(self.state.cut_motor_speed or DEFAULT_CUT_MOTOR_SPEED)}", "设置电锯速度"),
                (f"CFG PUMPSPEED {int(self.state.pump_cut_speed if self.state.pump_cut_speed is not None else DEFAULT_PUMP_CUT_SPEED)}", "设置水泵速度"),
                (f"CFG CUTTIME {int(self.state.cut_time_ms or DEFAULT_CUT_TIME_MS)}", "设置终点停留延时"),
                (f"CFG OFFSET {float(self.state.blade_offset_mm)}", "设置刀片物理偏移量"),
                (f"CFG SERVOCENTER {int(self.state.servo_center or DEFAULT_SERVO_CENTER)}", "设置舵机对接位"),
                (f"CFG SERVOHEAD {int(self.state.servo_head or DEFAULT_SERVO_HEAD)}", "设置鱼头分拣位"),
                (f"CFG SERVOBODY {int(self.state.servo_body or DEFAULT_SERVO_BODY)}", "设置鱼段分拣位"),
            ]
            for cmd, label in cfg_items:
                ok = self._send_command_checked(cmd, ("CFG_OK",), label, side=side)
                if not ok:
                    self.serial.send_cmd("JOBCLEAR", wait_prefixes=("JOBCLEARED",), timeout=1.5)
                    self.state.job_task_sent = False
                    self.state.job_running = False
                    self.state.sequence_running = False
                    self.state.auto_state_text = "参数配置失败，请检查日志后重试。"
                    return False
            if start_now:
                ok = self._send_command_checked("JOBSTART", ("JOBSTATE ",), "启动任务", timeout=4.0, side=side)
                if ok:
                    self.state.job_running = True
                    self.state.sequence_running = True
                    self.state.job_task_sent = True
                    self.state.job_state_text = "任务已下发并启动"
                    self.state.auto_state_text = "Arduino 正在执行整鱼加工任务。"
                    self.state.job_progress_text = f"总刀数：{len(self.state.generated_tasks)}"
                    self.add_log(self._format_side_log(side, "整鱼加工任务已下发并启动。"))
                    keep_lock_for_running_job = True
                    return True
            else:
                self.state.job_task_sent = True
                self.state.job_state_text = "任务已下发，等待启动"
                self.add_log(self._format_side_log(side, "刀长任务表已下发到 Arduino。"))
                return True
            self.state.job_task_sent = False
            return False
        finally:
            if not keep_lock_for_running_job:
                self.state.op_lock.release()

    def query_limits(self, side: str = "系统"):
        if self.serial is not None:
            self.serial.send_cmd("LIMITS")

    def cut_test(self, side: str = "系统"):
        if self._manual_action_locked("横切测试", side=side):
            return
        self._send_locked_manual_command("CUTTEST", "横切测试", "开始执行丝杠2闭环横切测试。", side=side)

    def cut_home(self, side: str = "系统"):
        if self._manual_action_locked("刀头复位", side=side):
            return
        self._send_locked_manual_command("CUTHOME", "刀头复位", "开始让丝杠2退回起点传感器。", side=side)

    def cut_forward(self, side: str = "系统"):
        if self._manual_action_locked("刀头正向推进", side=side):
            return
        self._send_locked_manual_command("CUTFORWARD", "刀头正向推进", "开始让丝杠2向右盲走 10mm 寸动。", side=side)

    def cut_reverse(self, side: str = "系统"):
        if self._manual_action_locked("刀头反向回退", side=side):
            return
        self._send_locked_manual_command("CUTREVERSE", "刀头反向回退", "开始让丝杠2向左盲走 10mm 寸动。", side=side)

    def query_job_status(self, side: str = "系统"):
        if self.serial is not None:
            self.serial.send_cmd("JOBSTATUS")

    def reconnect_serial(self, side: str = "系统"):
        if self.serial is None:
            return False
        self.add_log(self._format_side_log(side, "正在重连硬件。"))
        return self.serial.reconnect()

    def home_stepper(self, side: str = "系统"):
        if self._manual_action_locked("丝杠回零", side=side):
            return
        self._send_locked_manual_command("HOME", "丝杠回零", "丝杠开始物理回零。", side=side)

    def feed_until_stop(self, side: str = "系统"):
        if self._manual_action_locked("送料到挡板", side=side):
            return
        self._send_locked_manual_command("FEEDSTOP", "送料到挡板", "开始送料到挡板限位。", side=side)

    def eject_tail(self, side: str = "系统"):
        if self._manual_action_locked("尾料排空", side=side):
            return
        eject_ms = int(self.state.eject_ms or DEFAULT_EJECT_MS)
        self._send_locked_manual_command(
            f"EJECT {eject_ms}",
            "尾料排空",
            f"开始尾料排空，持续 {eject_ms} ms。",
            side=side,
        )

    def run_all_motors(self, speed: int, side: str = "系统"):
        if self._manual_action_locked("控制送料电机", side=side):
            return
        speed = int(max(-100, min(100, speed)))
        self.state.all_motor_speed = speed
        self.state.motor1_speed = speed
        self.state.motor2_speed = speed
        self._send_locked_manual_command(
            f"ALL {speed}",
            "控制送料电机",
            f"送料电机统一速度设置为 {speed}。",
            side=side,
        )

    def send_motor_cmd(self, motor_index: int, speed: int, side: str = "系统"):
        if self._manual_action_locked(f"控制电机 M{motor_index}", side=side):
            return
        speed = int(max(-100, min(100, speed)))
        if motor_index == 1:
            self.state.motor1_speed = speed
        elif motor_index == 2:
            self.state.motor2_speed = speed
        if self.state.motor1_speed == self.state.motor2_speed:
            self.state.all_motor_speed = self.state.motor1_speed
        self._send_locked_manual_command(
            f"M{motor_index} {speed}",
            f"控制电机 M{motor_index}",
            f"电机 M{motor_index} 速度设置为 {speed}。",
            side=side,
        )

    def set_servo_angle(self, angle: int, side: str = "系统"):
        if self._manual_action_locked("转动舵机", side=side):
            return
        angle = int(max(0, min(180, angle)))
        self.state.servo_angle_deg = angle
        self._send_locked_manual_command(f"FLIP {angle}", "转动舵机", f"舵机已转到 {angle} 度。", side=side)

    def set_servo2_angle(self, angle: int, side: str = "系统"):
        if self._manual_action_locked("转动舵机2", side=side):
            return
        angle = int(max(0, min(180, angle)))
        self.state.servo2_angle_deg = angle
        self._send_locked_manual_command(f"FLIP2 {angle}", "转动舵机2", f"舵机2已转到 {angle} 度。", side=side)

    def send_servo_named(self, name: str, side: str = "系统"):
        if self._manual_action_locked("切换分拣舵机", side=side):
            return
        alias = {"CENTER": "对接位", "HEAD": "鱼头位", "BODY": "鱼段位"}.get(name, name)
        preset_angle = {"CENTER": self.state.servo_center, "HEAD": self.state.servo_head, "BODY": self.state.servo_body}.get(name)
        if preset_angle is not None:
            self.state.servo_angle_deg = int(preset_angle)
        self._send_locked_manual_command(f"SERVO {name}", "切换分拣舵机", f"舵机切换到{alias}。", side=side)

    def move_stepper(self, distance_mm: float, side: str = "系统"):
        if self._manual_action_locked("移动丝杠", side=side):
            return
        distance_mm = float(distance_mm)
        target_pos_mm = self.state.current_pos_mm + distance_mm
        if target_pos_mm < 0.0 or target_pos_mm > STEPPER_SOFT_LIMIT_MM:
            self.add_log(
                self._format_side_log(
                    side,
                    f"丝杆移动失败：目标坐标 {target_pos_mm:.2f} mm 超出安全范围 0.00~{STEPPER_SOFT_LIMIT_MM:.2f} mm。"
                )
            )
            self.state.op_lock.release()
            return
        self.state.step_dist_mm = distance_mm
        self._send_locked_manual_command(
            f"STEP {distance_mm:.2f}",
            "移动丝杠",
            f"丝杆移动 {distance_mm:.2f} mm。",
            side=side,
        )

    def set_stepper_zero(self, side: str = "系统"):
        if self._manual_action_locked("设置丝杠原点", side=side):
            return
        self._send_locked_manual_command("SETZERO", "设置丝杠原点", "当前丝杆位置已设为原点。", side=side)

    def goto_stepper_zero(self, side: str = "系统"):
        if self._manual_action_locked("返回设定零点", side=side):
            return
        self._send_locked_manual_command("GOTOZERO", "返回设定零点", "丝杠返回软件标定零点。", side=side)

    def query_weight(self, side: str = "系统"):
        if self._manual_action_locked("读取重量", side=side):
            return
        try:
            with self.state.weight_poll_lock:
                weight_g = self.query_weight_g(log_on_fail=True)
            if weight_g is not None:
                self.add_log(self._format_side_log(side, f"当前重量：{self._format_weight_text(weight_g)}。"))
        finally:
            self.state.op_lock.release()

    def tare_weight_sensor(self, side: str = "系统"):
        if self._manual_action_locked("称重去皮", side=side):
            return
        try:
            if self.serial is None:
                self.add_log(self._format_side_log(side, "称重去皮失败：Arduino 未连接。"))
                return
            with self.state.weight_poll_lock:
                line = self.serial.send_cmd_get_line(
                    "TARE",
                    wait_prefixes=("TARE_OK", "WEIGHT_ERR"),
                    reject_prefixes=("ERR ", "JOBERROR", "STOPPED"),
                    timeout=5.0,
                )
            if line and line.startswith("TARE_OK"):
                self.state.current_weight_g = 0.0
                self.state.last_weight_at = time.time()
                self.add_log(self._format_side_log(side, "已发送称重去皮命令。"))
            else:
                self.add_log(self._format_side_log(side, "称重去皮失败，请检查称重模块接线、供电和当前载荷。"))
        finally:
            self.state.op_lock.release()

    def emergency_stop(self, side: str = "系统"):
        self.state.sequence_running = False
        self.state.job_running = False
        self.state.target_present_latch = False
        self.state.stable_count = 0
        self.state.last_stable_length = None
        self.state.absent_count = 0
        self.state.auto_state_text = "已触发急停，等待恢复。"
        self.state.estop_weight_suppress_until = time.time() + 6.0
        self.state.all_motor_speed = 0
        self.state.motor1_speed = 0
        self.state.motor2_speed = 0
        self.state.op_lock.release()
        if self.serial is not None:
            self.serial.send_cmd("STOP")
            self.serial.send_cmd("ALL 0")
        if self.vision:
            self.vision.detection_paused = False
        self.add_log(self._format_side_log(side, "全局急停已触发，所有动作立即中断。操作权已释放。"))
    def take_snapshot(self, side: str = "系统"):
        if self.state.snapshot_running:
            self.add_log(self._format_side_log(side, "手动测长正在执行中，请稍候。"))
            return
        if self.state.job_running:
            self.add_log(self._format_side_log(side, "当前 Arduino 正在执行加工任务，暂不允许手动测长。"))
            return
        if not self.state.op_lock.acquire(side):
            holder = self.state.op_lock.holder()
            self.add_log(self._format_side_log(side, f"操作权被【{holder}端】持有，连拍测长已拦截。"))
            return
        if not self.vision or not self.vision.vision_enabled:
            self.add_log(self._format_side_log(side, "拍照失败：视觉尚未启动。"))
            self.state.op_lock.release()
            return
        frame, _ = self.vision.get_latest_frame_copy()
        if frame is None:
            self.add_log(self._format_side_log(side, "拍照失败：当前没有可用画面。"))
            self.state.op_lock.release()
            return
        threading.Thread(target=self._snapshot_task, args=(False,), daemon=True).start()

    def _auto_measure_and_maybe_start(self):
        if self.state.snapshot_running or self.state.job_running:
            return
        if not self.state.op_lock.acquire("自动"):
            holder = self.state.op_lock.holder()
            self.state.auto_state_text = f"操作权被【{holder}端】持有，本次自动测长已跳过。"
            return
        threading.Thread(target=self._snapshot_task, args=(True,), daemon=True).start()

    def _recommended_feed_distance(self, length_cm: float) -> float:
        return round((length_cm * 10.0) / 16.0, 2)

    def _snapshot_task(self, auto_after_measure: bool):
        snapshot_lock_acquired = False
        lock_handed_to_running_job = False
        try:
            if self.vision is None:
                return
            snapshot_lock_acquired = self.vision._snapshot_lock.acquire(blocking=False)
            if not snapshot_lock_acquired:
                self.add_log("连拍测长正在执行中，请勿重复触发。")
                return
            self.state.snapshot_running = True
            if auto_after_measure:
                self.state.sequence_running = True
            self.vision.ensure_model()
            self.state.general_status = f"正在连拍 {BURST_CAPTURE_COUNT} 张并识别..."
            self.state.live_detail_text = "正在执行后台连拍识别，请保持带鱼姿态不动。"
            frames = self.vision.collect_burst_frames(BURST_CAPTURE_COUNT, BURST_CAPTURE_INTERVAL)
            if not frames:
                self.add_log("连拍失败：当前没有拿到有效画面。")
                self.state.general_status = "连拍失败：没有采集到画面。"
                if auto_after_measure:
                    self.state.sequence_running = False
                return
            result = self.vision.analyze_burst_frames(frames)
            if not result["detected"] or not result["best_result"]:
                self.state.last_snapshot_length = 0.0
                self.state.last_snapshot_head_length = 0.0
                self.state.last_snapshot_body_length = 0.0
                self.state.last_snapshot_time = time.time()
                self.state.live_result_text = "未识别"
                self.state.live_detail_text = f"本次连拍 {result['total_count']} 张，但没有识别到带鱼。"
                self.state.general_status = "连拍完成，但未识别到目标。"
                self.add_log(f"本次有效 0/{result['total_count']} 张，最终未识别出带鱼。")
                if auto_after_measure:
                    self.state.sequence_running = False
                    self.state.target_present_latch = False
                return
            best_result = result["best_result"]
            final_length = result["final_length"]
            final_head_length = result.get("final_head_length", 0.0)
            final_body_length = result.get("final_body_length", 0.0)
            valid_count = result["valid_count"]
            total_count = result["total_count"]
            success, jpeg_img = cv2.imencode(".jpg", best_result["annotated"], [cv2.IMWRITE_JPEG_QUALITY, SNAPSHOT_QUALITY])
            if success:
                self.state.last_snapshot_jpeg = jpeg_img.tobytes()
            os.makedirs(SNAPSHOT_DIR, exist_ok=True)
            filename = f"snapshot_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
            snapshot_path = os.path.join(SNAPSHOT_DIR, filename)
            cv2.imwrite(snapshot_path, best_result["annotated"], [cv2.IMWRITE_JPEG_QUALITY, SNAPSHOT_QUALITY])
            self.state.snapshots_taken += 1
            self.state.last_snapshot_time = time.time()
            self.state.last_snapshot_path = snapshot_path
            self.state.last_snapshot_length = final_length
            self.state.last_snapshot_head_length = final_head_length
            self.state.last_snapshot_body_length = final_body_length
            self.state.step_dist_mm = self._recommended_feed_distance(final_length)
            self.state.last_detected_length = final_length
            self.state.last_detection_at = self.state.last_snapshot_time
            with self.state.weight_poll_lock:
                weight_g = self.query_weight_g(log_on_fail=True)
            if weight_g is None:
                weight_g = self.state.current_weight_g
            if weight_g is not None:
                self.state.last_snapshot_weight_g = weight_g
                self.state.current_weight_g = weight_g
                self.state.last_weight_at = time.time()
            else:
                self.state.last_snapshot_weight_g = 0.0
            self.state.live_result_text = f"{final_length:.1f} cm"
            voice_list = ["changdu.wav"] + self._number_to_wavs(final_length) + ["limi.wav"]
            if weight_g is not None and weight_g > 0:
                if weight_g >= 1000:
                    voice_list += ["zhongliang.wav"] + self._number_to_wavs(weight_g / 1000.0) + ["qianke.wav"]
                else:
                    voice_list += ["zhongliang.wav"] + self._number_to_wavs(weight_g) + ["ke.wav"]
            self.play_voice_sequence(voice_list)
            self.state.live_detail_text = f"鱼头 {final_head_length:.1f} cm，鱼身 {final_body_length:.1f} cm，本次有效 {valid_count}/{total_count} 张。"
            self.state.general_status = "连拍识别完成。"
            self.record_measurement(
                final_length,
                "连拍快照" if not auto_after_measure else "自动测长",
                f"有效 {valid_count}/{total_count} 张",
            )
            self.add_log(f"本次有效 {valid_count}/{total_count} 张，最终长度 {final_length:.2f} cm。")
            task_side = "自动" if auto_after_measure else "系统"
            self.generate_tasks_from_latest(side=task_side)
            if auto_after_measure:
                if self.dispatch_and_start_job(side="自动"):
                    self.state.last_trigger_time = time.time()
                    self.state.target_present_latch = True
                    lock_handed_to_running_job = True
                else:
                    self.state.sequence_running = False
                    self.state.target_present_latch = False
        except Exception as e:
            self.add_log(f"连拍测长异常：{e}")
            self.state.general_status = f"连拍测长异常：{e}"
            if auto_after_measure:
                self.state.sequence_running = False
                self.state.target_present_latch = False
        finally:
            self.state.snapshot_running = False
            if snapshot_lock_acquired and self.vision is not None:
                self.vision._snapshot_lock.release()
            if not lock_handed_to_running_job:
                self.state.op_lock.release()

    def on_detection_event(self, detected: bool, max_len_cm: float):
        now = time.time()
        if detected:
            self.state.last_detected_length = max_len_cm
            self.state.last_detection_at = now
            self.state.live_result_text = f"{max_len_cm:.1f} cm"
            self.state.live_detail_text = "检测到目标，正在后台低频跟踪稳定长度。"
            self.record_measurement(max_len_cm, "实时识别", "自动模式低频测长")
        else:
            self.state.live_detail_text = "自动模式中：等待目标进入画面。"
        if not self.state.auto_enabled or self.state.sequence_running or self.state.snapshot_running or self.state.job_running:
            return
        tolerance = float(self.state.tolerance or DEFAULT_TOLERANCE)
        absent_limit = int(self.state.absent_frames or DEFAULT_ABSENT_FRAMES)
        cooldown = float(self.state.cooldown or DEFAULT_COOLDOWN)
        if detected:
            self.state.absent_count = 0
            if self.state.target_present_latch:
                self.state.auto_state_text = "目标已锁定，等待当前整鱼流程结束。"
                return
            if self.state.last_stable_length is None:
                self.state.stable_count = 1
                self.state.last_stable_length = max_len_cm
            elif abs(max_len_cm - self.state.last_stable_length) <= tolerance:
                self.state.stable_count += 1
            else:
                self.state.stable_count = 1
                self.state.last_stable_length = max_len_cm
            self.state.auto_state_text = f"自动锁定中：{self.state.stable_count}/{int(self.state.stable_frames)}"
            if self.state.stable_count >= int(self.state.stable_frames) and (now - self.state.last_trigger_time) >= cooldown:
                self.state.auto_state_text = "已锁定稳定目标，准备自动测长并下发加工。"
                self._auto_measure_and_maybe_start()
        else:
            self.state.stable_count = 0
            self.state.last_stable_length = None
            self.state.absent_count += 1
            if self.state.absent_count >= absent_limit:
                self.state.target_present_latch = False
                self.state.auto_state_text = "等待目标进入画面。"

    def record_measurement(self, length_cm: float, source: str, note: str):
        if length_cm <= 0:
            return
        now = time.time()
        if source == "实时识别":
            recently_recorded = now - self.state.last_history_record_at < 1.5
            similar_length = self.state.last_history_length is not None and abs(length_cm - self.state.last_history_length) < 0.4
            if recently_recorded and similar_length:
                return
        self.state.last_history_record_at = now
        self.state.last_history_length = length_cm
        self.state.measurement_rows.insert(
            0,
            {
                "time": time.strftime("%H:%M:%S", time.localtime(now)),
                "length": f"{length_cm:.1f} cm",
                "source": source,
                "note": note,
            },
        )
        self.state.measurement_rows = self.state.measurement_rows[:MAX_MEASUREMENT_ROWS]
        self.state.record_count += 1
        self.state.length_sum += length_cm
        self.state.max_detected_length = max(self.state.max_detected_length, length_cm)

    def get_status_payload(self) -> Dict[str, Any]:
        average_length = self.state.length_sum / self.state.record_count if self.state.record_count else 0.0
        length_trend_points = self._build_length_trend_points()
        payload = {
            "serial_connected": self.state.serial_connected,
            "serial_port": self.state.serial_port,
            "auto_enabled": self.state.auto_enabled,
            "vision_running": self.state.vision_running,
            "live_overlay_enabled": self.state.live_overlay_enabled,
            "general_status": self.state.general_status,
            "live_result_text": self.state.live_result_text,
            "live_detail_text": self.state.live_detail_text,
            "auto_state_text": self.state.auto_state_text,
            "current_weight_g": self.state.current_weight_g,
            "last_weight_at": self.state.last_weight_at,
            "job_running": self.state.job_running,
            "job_state_text": self.state.job_state_text,
            "job_progress_text": self.state.job_progress_text,
            "job_result_text": self.state.job_result_text,
            "current_pos_mm": self.state.current_pos_mm,
            "cpu_percent": self.state.cpu_percent,
            "mem_percent": self.state.mem_percent,
            "cpu_temp": self.state.cpu_temp,
            "all_motor_speed": self.state.all_motor_speed,
            "motor1_speed": self.state.motor1_speed,
            "motor2_speed": self.state.motor2_speed,
            "servo_angle_deg": self.state.servo_angle_deg,
            "servo2_angle_deg": self.state.servo2_angle_deg,
            "last_detected_length": self.state.last_detected_length,
            "last_detection_at": self.state.last_detection_at,
            "last_snapshot_length": self.state.last_snapshot_length,
            "last_snapshot_head_length": self.state.last_snapshot_head_length,
            "last_snapshot_body_length": self.state.last_snapshot_body_length,
            "last_snapshot_time": self.state.last_snapshot_time,
            "limit_home_state": self.state.limit_home_state,
            "limit_stop_state": self.state.limit_stop_state,
            "cut_start_state": self.state.cut_start_state,
            "cut_end_state": self.state.cut_end_state,
            "arduino_ready": self.state.arduino_ready,
            "generated_tasks": list(self.state.generated_tasks),
            "generated_tasks_count": len(self.state.generated_tasks),
            "current_total_mm": self.state.current_total_mm,
            "current_head_mm": self.state.current_head_mm,
            "current_remain_mm": self.state.current_remain_mm,
            "measurement_rows": list(self.state.measurement_rows),
            "last_serial_line": self.state.last_serial_line,
            "snapshots_taken": self.state.snapshots_taken,
            "auto_runs_count": self.state.auto_runs_count,
            "session_started_at": self.state.session_started_at,
            "record_count": self.state.record_count,
            "average_length": average_length,
            "max_detected_length": self.state.max_detected_length,
            "op_lock_holder": self.state.op_lock.holder(),
            "cut_mode": self.state.cut_mode,
            "voice_broadcast": self.state.voice_broadcast,
            "fixed_length_mm": self.state.fixed_length_mm,
            "avg_parts_count": self.state.avg_parts_count,
            "auto_speed": self.state.auto_speed,
            "stable_frames": self.state.stable_frames,
            "absent_frames": self.state.absent_frames,
            "tolerance": self.state.tolerance,
            "cooldown": self.state.cooldown,
            "settle_ms": self.state.settle_ms,
            "sort_hold_ms": self.state.sort_hold_ms,
            "center_hold_ms": self.state.center_hold_ms,
            "eject_ms": self.state.eject_ms,
            "cut_time_ms": self.state.cut_time_ms,
            "servo_center": self.state.servo_center,
            "servo_head": self.state.servo_head,
            "servo_body": self.state.servo_body,
            "servo2_down": self.state.servo2_down,
            "servo2_up": self.state.servo2_up,
            "cut_motor_speed": self.state.cut_motor_speed,
            "pump_cut_speed": self.state.pump_cut_speed,
            "blade_offset_mm": self.state.blade_offset_mm,
            "step_dist_mm": self.state.step_dist_mm,
            "length_trend_points": length_trend_points,
            "recent_logs": list(self.state.pending_logs),
        }
        return payload

    def shutdown(self):
        self.state.weight_poll_running = False
        self.state.serial_reconnect_running = False
        self.emergency_stop()
        if self.vision:
            self.vision.stop_vision()
        if self.serial:
            self.serial.close()


global_state = SystemState()
app_controller = AppController(global_state)
app = FastAPI(title="Fish Workstation Backend")
api_router = APIRouter(prefix="/api")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def index_page():
    return FileResponse(os.path.join(BACKEND_DIR, "index.html"))


@app.get("/pc")
def pc_page():
    return FileResponse(os.path.join(BACKEND_DIR, "pc.html"))


@api_router.get("/status")
def get_status():
    return app_controller.get_status_payload()


@api_router.get("/latest_snapshot")
def get_latest_snapshot():
    jpeg = app_controller.state.last_snapshot_jpeg or LATEST_SNAPSHOT_PLACEHOLDER
    if not jpeg:
        return Response(status_code=404)
    return Response(
        content=jpeg,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
    )


@api_router.post("/action/snapshot")
def action_snapshot(client: str = "系统"):
    app_controller.take_snapshot(side=client)
    return {"status": "ok"}


@api_router.post("/action/start_vision")
def action_start_vision(client: str = "系统"):
    app_controller.start_vision(side=client)
    return {"status": "ok"}


@api_router.post("/action/stop_vision")
def action_stop_vision(client: str = "系统"):
    app_controller.stop_vision(side=client)
    return {"status": "ok"}


@api_router.post("/action/set_live_overlay")
def action_set_live_overlay(payload: Dict[str, Any] = Body(default={}), client: str = "系统"):
    app_controller.set_live_overlay(bool(payload.get("enabled", False)), side=client)
    return {"status": "ok"}


@api_router.post("/action/generate_tasks")
def action_generate_tasks(client: str = "系统"):
    app_controller.generate_tasks_from_latest(side=client)
    return {"status": "ok"}


@api_router.post("/action/query_weight")
def action_query_weight(client: str = "系统"):
    app_controller.query_weight(side=client)
    return {"status": "ok"}


@api_router.post("/action/tare")
def action_tare(client: str = "系统"):
    app_controller.tare_weight_sensor(side=client)
    return {"status": "ok"}


@api_router.post("/action/query_limits")
def action_query_limits(client: str = "系统"):
    app_controller.query_limits(side=client)
    return {"status": "ok"}


@api_router.post("/action/cut_test")
def action_cut_test(client: str = "系统"):
    app_controller.cut_test(side=client)
    return {"status": "ok"}


@api_router.post("/action/cut_home")
def action_cut_home(client: str = "系统"):
    app_controller.cut_home(side=client)
    return {"status": "ok"}


@api_router.post("/action/cut_forward")
def action_cut_forward(client: str = "系统"):
    app_controller.cut_forward(side=client)
    return {"status": "ok"}


@api_router.post("/action/cut_reverse")
def action_cut_reverse(client: str = "系统"):
    app_controller.cut_reverse(side=client)
    return {"status": "ok"}


@api_router.post("/action/query_job_status")
def action_query_job_status(client: str = "系统"):
    app_controller.query_job_status(side=client)
    return {"status": "ok"}


@api_router.post("/action/reconnect")
def action_reconnect(client: str = "系统"):
    app_controller.reconnect_serial(side=client)
    return {"status": "ok"}


@api_router.post("/action/clear_tasks")
def action_clear_tasks(client: str = "系统"):
    app_controller.clear_generated_tasks(side=client)
    return {"status": "ok"}


@api_router.post("/action/clear_measurements")
def action_clear_measurements(client: str = "系统"):
    app_controller.clear_measurements(side=client)
    return {"status": "ok"}


@api_router.post("/action/reset_config")
def action_reset_config(client: str = "系统"):
    app_controller.reset_config(side=client)
    return {"status": "ok"}


@api_router.post("/action/home_stepper")
def action_home_stepper(client: str = "系统"):
    app_controller.home_stepper(side=client)
    return {"status": "ok"}


@api_router.post("/action/feed_until_stop")
def action_feed_until_stop(client: str = "系统"):
    app_controller.feed_until_stop(side=client)
    return {"status": "ok"}


@api_router.post("/action/eject_tail")
def action_eject_tail(client: str = "系统"):
    app_controller.eject_tail(side=client)
    return {"status": "ok"}


@api_router.post("/action/set_stepper_zero")
def action_set_stepper_zero(client: str = "系统"):
    app_controller.set_stepper_zero(side=client)
    return {"status": "ok"}


@api_router.post("/action/goto_zero")
def action_goto_zero(client: str = "系统"):
    app_controller.goto_stepper_zero(side=client)
    return {"status": "ok"}


@api_router.post("/action/move_stepper")
def action_move_stepper(payload: Dict[str, Any] = Body(default={}), client: str = "系统"):
    app_controller.move_stepper(float(payload.get("distance_mm", 0.0)), side=client)
    return {"status": "ok"}


@api_router.post("/action/set_servo_angle")
def action_set_servo_angle(payload: Dict[str, Any] = Body(default={}), client: str = "系统"):
    app_controller.set_servo_angle(int(payload.get("angle", 90)), side=client)
    return {"status": "ok"}


@api_router.post("/action/set_servo2_angle")
def action_set_servo2_angle(payload: Dict[str, Any] = Body(default={}), client: str = "系统"):
    app_controller.set_servo2_angle(int(payload.get("angle", DEFAULT_SERVO2_UP)), side=client)
    return {"status": "ok"}


@api_router.post("/action/send_servo_named")
def action_send_servo_named(payload: Dict[str, Any] = Body(default={}), client: str = "系统"):
    app_controller.send_servo_named(str(payload.get("name", "CENTER")), side=client)
    return {"status": "ok"}


@api_router.post("/action/run_all_motors")
def action_run_all_motors(payload: Dict[str, Any] = Body(default={}), client: str = "系统"):
    app_controller.run_all_motors(int(payload.get("speed", 0)), side=client)
    return {"status": "ok"}


@api_router.post("/action/send_motor")
def action_send_motor(payload: Dict[str, Any] = Body(default={}), client: str = "系统"):
    app_controller.send_motor_cmd(
        int(payload.get("motor_index", 1)),
        int(payload.get("speed", 0)),
        side=client,
    )
    return {"status": "ok"}


@api_router.post("/config/update")
def config_update(payload: Dict[str, Any] = Body(default={}), client: str = "系统"):
    app_controller.update_config(payload, side=client)
    return {"status": "ok"}


@api_router.post("/action/dispatch_start")
def action_dispatch_start(client: str = "系统"):
    app_controller.dispatch_and_start_job(side=client)
    return {"status": "ok"}


@api_router.post("/action/dispatch_only")
def action_dispatch_only(client: str = "系统"):
    app_controller.dispatch_job_only(side=client)
    return {"status": "ok"}


@api_router.post("/action/estop")
def action_estop(client: str = "系统"):
    app_controller.emergency_stop(side=client)
    return {"status": "ok"}


async def mjpeg_frame_generator():
    while True:
        vision = app_controller.vision
        if vision is None or not vision.vision_enabled:
            await asyncio.sleep(0.05)
            continue

        jpeg = vision.latest_jpeg
        if not jpeg:
            await asyncio.sleep(0.03)
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
        )
        await asyncio.sleep(0.10 if app_controller.state.live_overlay_enabled else 0.03)


@api_router.get("/video_feed")
def video_feed():
    return StreamingResponse(
        mjpeg_frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


app.include_router(api_router)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
