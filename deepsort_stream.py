#!/usr/bin/env python3
import os
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'

import asyncio
import contextlib
import signal
import time
import json
import threading
from dataclasses import dataclass
from datetime import datetime

import websockets
import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

Gst.init(None)

# ---- FIFO 설정 (CAN 전송용) ----
FIFO_PATH = "/tmp/drone_coords"

HOST = "0.0.0.0"
PORT = 3000

PATH_GIMBAL = "/gimbal"  # 드론 탐지 영상 (IMX477)
PATH_FRONT  = "/front"   # USB 카메라 (원본)

SEND_TIMEOUT_SEC = 0.05

TARGET_FPS_GIMBAL = 30
TARGET_FPS_FRONT  = 30

JPEG_QUALITY_GIMBAL = 50  # 바운딩 박스 포함 영상
JPEG_QUALITY_FRONT  = 80  # USB 원본

# ---- USB 카메라 (원본, 변경 없음) ----
PIPELINE_FRONT = f"""
v4l2src device=/dev/cam_front !
video/x-raw,framerate=30/1 !
videoconvert !
jpegenc quality={JPEG_QUALITY_FRONT} !
appsink name=out emit-signals=false sync=false max-buffers=2 drop=true
""".strip()

print("="*50)
print("Drone Tracking System - WebSocket (YOLO + DeepSORT)")
print("="*50)

# YOLO 모델 로딩
print("\nLoading TensorRT model...")
model = YOLO('/home/wooyeong/PBL/yolo/best3.engine')
print("TensorRT model loaded!")

TARGET_CLASS_ID = 0
print(f"Detecting class ID: {TARGET_CLASS_ID}")

# DeepSORT 초기화
tracker = DeepSort(
    max_age=60,
    n_init=1,
    max_iou_distance=0.9,
    nms_max_overlap=0.9,
    max_cosine_distance=0.7,
    embedder="mobilenet"
)
'''
# 색상 보정 함수
def color_correction(frame):
    # RGB 채널 조정
    b, g, r = cv2.split(frame)
    r = np.clip(r.astype(np.float32) * 1.20, 0, 255).astype(np.uint8)
    g = np.clip(g.astype(np.float32) * 1.20, 0, 255).astype(np.uint8)
    b = np.clip(b.astype(np.float32) * 1.20, 0, 255).astype(np.uint8)
    frame = cv2.merge([b, g, r])
    
    # HSV 조정 (Hue Shift)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] - 28.8) % 180  # -16 * 180 / 100 = -28.8
    frame = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    return frame
'''
# ID 관리 클래스
class IDManager:
    def __init__(self):
        self.next_id = 1
        self.active_ids = set()
    
    def get_id(self, deepsort_id, deepsort_to_our_id):
        if deepsort_id in deepsort_to_our_id:
            return deepsort_to_our_id[deepsort_id]
        new_id = self.next_id
        self.next_id += 1
        self.active_ids.add(new_id)
        deepsort_to_our_id[deepsort_id] = new_id
        return new_id
    
    def remove_id(self, our_id):
        if our_id in self.active_ids:
            self.active_ids.remove(our_id)
        if len(self.active_ids) == 0:
            self.next_id = 1
            print("[ID Manager] All drones lost - resetting IDs to 1")

# 확장 메모리
class ExtendedTracker:
    def __init__(self, memory_time=5.0):
        self.memory_time = memory_time
        self.lost_tracks = {}
    
    def update_lost(self, track_id, bbox, velocity, timestamp):
        self.lost_tracks[track_id] = {
            'last_pos': np.array([(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2]),
            'last_vel': velocity,
            'last_seen': timestamp,
            'bbox_size': np.array([bbox[2]-bbox[0], bbox[3]-bbox[1]])
        }
    
    def remove_track(self, track_id):
        if track_id in self.lost_tracks:
            del self.lost_tracks[track_id]

# 초기화
id_manager = IDManager()
deepsort_to_our_id = {}
ext_tracker = ExtendedTracker(memory_time=10.0)
prev_tracks = {}

@dataclass
class Stats:
    frames: int = 0
    dropped_send: int = 0
    send_fail: int = 0
    sample_fail: int = 0

# ---- IMX477 드론 탐지 브로드캐스터 (OpenCV 기반) ----
class DroneDetectionBroadcaster:
    def __init__(self, name: str, target_fps: int):
        self.name = name
        self.target_fps = target_fps
        
        self.cap = None
        self.latest = None
        self.latest_confidence = 0
        self.stats = Stats()
        
        # ✅ FPS 계산용
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0
        
        # CAN 전송용 좌표 저장
        self.latest_err_x = 0
        self.latest_err_y = 0
        self._coord_lock = threading.Lock()
        
        # FIFO 관련
        self._fifo_fd = None
        self._fifo_stop = False
        self._fifo_thread = None
        
        self._stop = asyncio.Event()
        self._frame_event = asyncio.Event()
        
        # GStreamer 파이프라인
        self.pipeline = (
            "nvarguscamerasrc sensor-id=0 "
            "wbmode=1 "
            "exposurecompensation=0.5 ! "
            "video/x-raw(memory:NVMM), width=1920, height=1080, framerate=30/1 ! "
            "nvvidconv ! "
            "video/x-raw, width=1920, height=1080, format=BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=BGR ! "
            "appsink drop=true sync=false"
        )
        
        # 전역 변수 참조
        self.id_manager = id_manager
        self.deepsort_to_our_id = deepsort_to_our_id
        self.ext_tracker = ext_tracker
        self.prev_tracks = prev_tracks
        
    def start_camera(self):
        print(f"[{self.name}] Opening camera...")
        self.cap = cv2.VideoCapture(self.pipeline, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            raise RuntimeError(f"[{self.name}] Failed to open camera")
        
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[{self.name}] Camera opened: {self.width}x{self.height}")
    
    def stop_camera(self):
        if self.cap:
            self.cap.release()
    
    def start_fifo_sender(self):
        self._fifo_thread = threading.Thread(target=self._fifo_sender_loop, daemon=True)
        self._fifo_thread.start()
        print(f"[{self.name}] FIFO sender started -> {FIFO_PATH}")
    
    def stop_fifo_sender(self):
        self._fifo_stop = True
        if self._fifo_fd is not None:
            try:
                os.close(self._fifo_fd)
            except:
                pass
            self._fifo_fd = None
    
    def _fifo_sender_loop(self):
        """FIFO로 좌표를 주기적으로 전송하는 스레드"""
        while not self._fifo_stop:
            try:
                # FIFO 열기 (C 프로그램이 읽기 대기 중이어야 함)
                if self._fifo_fd is None:
                    print(f"[{self.name}] Opening FIFO (waiting for C program)...")
                    self._fifo_fd = os.open(FIFO_PATH, os.O_WRONLY)
                    print(f"[{self.name}] FIFO connected!")
                
                # 좌표 전송 (30ms 주기)
                with self._coord_lock:
                    x = self.latest_err_x
                    y = self.latest_err_y
                
                msg = f"{x},{y}\n".encode('utf-8')
                os.write(self._fifo_fd, msg)
                
                time.sleep(0.025)  # ~25ms (CAN 30ms 주기보다 조금 빠르게)
                
            except BrokenPipeError:
                print(f"[{self.name}] FIFO disconnected (C program stopped?)")
                if self._fifo_fd is not None:
                    try:
                        os.close(self._fifo_fd)
                    except:
                        pass
                    self._fifo_fd = None
                time.sleep(1)  # 재연결 대기
                
            except FileNotFoundError:
                # FIFO 파일이 아직 없음 (C 프로그램이 생성 안 함)
                time.sleep(1)
                
            except Exception as e:
                if not self._fifo_stop:
                    print(f"[{self.name}] FIFO error: {e}")
                time.sleep(1)
    
    def request_stop(self):
        self._stop.set()
        self._frame_event.set()
    
    async def capture_loop(self):
        """핵심: to_thread 제거, 동기 처리"""
        min_period = (1.0 / self.target_fps) if self.target_fps > 0 else 0.0
        last_t = 0.0
        
        loop = asyncio.get_running_loop()
        
        while not self._stop.is_set():
            if min_period:
                now = time.monotonic()
                dt = now - last_t
                if dt < min_period:
                    await asyncio.sleep(min_period - dt)
                last_t = time.monotonic()
            
            try:
                jpeg_bytes, confidence = await loop.run_in_executor(None, self._capture_and_process)
                
                if jpeg_bytes:
                    self.latest = jpeg_bytes
                    self.latest_confidence = int(confidence * 100)
                    self.stats.frames += 1
                    
                    # ✅ FPS 계산
                    self.fps_counter += 1
                    elapsed = time.time() - self.fps_start_time
                    if elapsed >= 1.0:
                        self.current_fps = self.fps_counter / elapsed
                        self.fps_counter = 0
                        self.fps_start_time = time.time()
                    
                    self._frame_event.set()
                else:
                    self.stats.sample_fail += 1
                    
            except Exception as e:
                print(f"[{self.name}] Capture error: {e}")
                self.stats.sample_fail += 1
        
        self._frame_event.set()
    
    def _capture_and_process(self) -> tuple:
        """동기 처리: 프레임 읽기 → YOLO → 그리기 → 인코딩"""
        # 1. 프레임 읽기
        success, frame = self.cap.read()
        if not success:
            return b"", 0.0
        
        # 2. YOLO + DeepSORT 처리 및 그리기
        processed_frame, max_conf = self._process_frame(frame)
        
        # 3. JPEG 인코딩
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY_GIMBAL]
        success, encoded = cv2.imencode('.jpg', processed_frame, encode_param)
        
        if not success:
            return b"", 0.0
        
        return encoded.tobytes(), max_conf
    
    def _process_frame(self, frame):
        """YOLO + DeepSORT 처리"""
        timestamp = datetime.now()
        
        # YOLO 탐지
        results = model.predict(
            frame,
            conf=0.3,
            iou=0.7,
            half=True,
            verbose=False
        )
        
        # 검출된 bbox 수집
        detected_bboxes = set()
        detection_confidences = {}
        
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id != TARGET_CLASS_ID:
                    continue
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                
                bbox_key = (x1, y1, x2, y2)
                detected_bboxes.add(bbox_key)
                detection_confidences[bbox_key] = conf
        
        # DeepSORT 포맷 변환
        detections_for_tracker = []
        for bbox_key in detected_bboxes:
            x1, y1, x2, y2 = bbox_key
            conf = detection_confidences[bbox_key]
            detections_for_tracker.append(([x1, y1, x2-x1, y2-y1], conf, 'target'))
        
        # DeepSORT 업데이트
        tracks = tracker.update_tracks(detections_for_tracker, frame=frame)
        
        current_our_ids = set()
        detection_count = 0
        max_confidence = 0.0
        
        # 트래킹 및 시각화
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            deepsort_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            
            # 실제 검출 확인
            is_detected_now = False
            matched_conf = 0.0
            
            for bbox_key in detected_bboxes:
                bx1, by1, bx2, by2 = bbox_key
                if abs(x1-bx1) < 20 and abs(y1-by1) < 20 and abs(x2-bx2) < 20 and abs(y2-by2) < 20:
                    is_detected_now = True
                    matched_conf = detection_confidences[bbox_key]
                    break
            
            if not is_detected_now:
                continue
            
            if matched_conf > max_confidence:
                max_confidence = matched_conf
            
            # ID 매핑
            our_id = self.id_manager.get_id(deepsort_id, self.deepsort_to_our_id)
            current_our_ids.add(our_id)
            
            self.ext_tracker.remove_track(our_id)
            
            # 속도 계산
            velocity = np.array([0.0, 0.0])
            if our_id in self.prev_tracks:
                prev_pos = self.prev_tracks[our_id]['pos']
                current_pos = np.array([(x1+x2)/2, (y1+y2)/2])
                dt = (timestamp - self.prev_tracks[our_id]['timestamp']).total_seconds()
                if dt > 0:
                    velocity = (current_pos - prev_pos) / dt
            
            self.prev_tracks[our_id] = {
                'pos': np.array([(x1+x2)/2, (y1+y2)/2]),
                'bbox': [x1, y1, x2, y2],
                'vel': velocity,
                'timestamp': timestamp
            }
            
            detection_count += 1
            
            # 바운딩 박스 그리기
            dx, dy = (x1 + x2) // 2, (y1 + y2) // 2
            err_x = dx - (self.width // 2)
            err_y_raw = dy - (self.height // 2)
            err_y = -err_y_raw  # ✅ Y값 부호 반전
            
            # ✅ 드론 좌표 로그 출력
            print(f"[Drone ID:{our_id}] X:{err_x:4d}, Y:{err_y:4d} (Conf: {matched_conf:.2f})")
            
            # CAN 전송용 좌표 저장
            with self._coord_lock:
                self.latest_err_x = err_x
                self.latest_err_y = err_y  # ✅ 부호 반전된 값 저장
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (dx, dy), 5, (0, 0, 255), -1)
            
            cv2.putText(frame, f"ID:{our_id} ({matched_conf:.2f})", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            info_text = f"X:{err_x:4d} Y:{err_y:4d}"  # ✅ 부호 반전된 Y값 표시
            cv2.putText(frame, info_text, (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # 사라진 드론 처리
        lost_our_ids = set(self.prev_tracks.keys()) - current_our_ids
        for our_id in lost_our_ids:
            if our_id in self.prev_tracks:
                info = self.prev_tracks[our_id]
                self.ext_tracker.update_lost(our_id, info['bbox'], info['vel'], info['timestamp'])
        
        # 만료된 드론 ID 회수
        expired_our_ids = []
        for our_id in list(self.ext_tracker.lost_tracks.keys()):
            info = self.ext_tracker.lost_tracks[our_id]
            dt = (timestamp - info['last_seen']).total_seconds()
            
            if dt > self.ext_tracker.memory_time:
                expired_our_ids.append(our_id)
                del self.ext_tracker.lost_tracks[our_id]
                if our_id in self.prev_tracks:
                    del self.prev_tracks[our_id]
        
        for our_id in expired_our_ids:
            self.id_manager.remove_id(our_id)
            for ds_id, mapped_id in list(self.deepsort_to_our_id.items()):
                if mapped_id == our_id:
                    del self.deepsort_to_our_id[ds_id]
                    break
        
        # 중심점 및 정보 표시
        cv2.circle(frame, (self.width // 2, self.height // 2), 5, (255, 0, 0), -1)
        
        # ✅ 실시간 FPS 표시
        cv2.putText(frame, f"FPS: {self.current_fps:.1f} (TensorRT)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        status_text = f"Active: {detection_count} | Memory: {len(self.ext_tracker.lost_tracks)}"
        cv2.putText(frame, status_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame, max_confidence

# ---- USB 카메라 브로드캐스터 (GStreamer 기반, 원본 유지) ----
class JpegBroadcaster:
    def __init__(self, name: str, pipeline_str: str, target_fps: int):
        self.name = name
        self.pipeline_str = pipeline_str
        self.target_fps = target_fps

        self.pipeline = None
        self.appsink = None

        self.latest = None
        self.stats = Stats()
        
        # ✅ FPS 계산용
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0

        self._stop = asyncio.Event()
        self._frame_event = asyncio.Event()

    def start_gst(self):
        self.pipeline = Gst.parse_launch(self.pipeline_str)
        self.appsink = self.pipeline.get_by_name("out")
        if self.appsink is None:
            raise RuntimeError(f"[{self.name}] appsink named 'out' not found")

        self.appsink.set_property("sync", False)
        self.appsink.set_property("max-buffers", 2)
        self.appsink.set_property("drop", True)

        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError(f"[{self.name}] Failed to set pipeline to PLAYING")

    def stop_gst(self):
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)

    def request_stop(self):
        self._stop.set()
        self._frame_event.set()

    async def capture_loop(self):
        min_period = (1.0 / self.target_fps) if self.target_fps > 0 else 0.0
        last_t = 0.0

        while not self._stop.is_set():
            if min_period:
                now = time.monotonic()
                dt = now - last_t
                if dt < min_period:
                    await asyncio.sleep(min_period - dt)
                last_t = time.monotonic()

            sample = await asyncio.to_thread(self._try_pull_sample, 200_000_000)
            if sample is None:
                continue

            jpeg_bytes = self._sample_to_bytes(sample)
            if not jpeg_bytes:
                self.stats.sample_fail += 1
                continue

            self.latest = jpeg_bytes
            self.stats.frames += 1
            
            # ✅ FPS 계산
            self.fps_counter += 1
            elapsed = time.time() - self.fps_start_time
            if elapsed >= 1.0:
                self.current_fps = self.fps_counter / elapsed
                self.fps_counter = 0
                self.fps_start_time = time.time()
            
            self._frame_event.set()

        self._frame_event.set()

    def _try_pull_sample(self, timeout_ns: int):
        try:
            return self.appsink.emit("try-pull-sample", timeout_ns)
        except Exception:
            return None

    def _sample_to_bytes(self, sample) -> bytes:
        buf = sample.get_buffer()
        ok, mapinfo = buf.map(Gst.MapFlags.READ)
        if not ok:
            return b""
        try:
            return bytes(mapinfo.data)
        finally:
            buf.unmap(mapinfo)

def get_ws_path(websocket) -> str:
    p = getattr(websocket, "path", None)
    if p:
        return p
    req = getattr(websocket, "request", None)
    if req is not None:
        return getattr(req, "path", "") or ""
    return ""

class RouterServer:
    def __init__(self, bc_gimbal, bc_front):
        self.bc_gimbal = bc_gimbal
        self.bc_front = bc_front

    async def ws_handler(self, websocket):
        path = get_ws_path(websocket)

        if path == PATH_GIMBAL:
            await self._stream_from_gimbal(websocket)
            return

        if path == PATH_FRONT:
            if self.bc_front is None:
                await websocket.close(code=1008, reason="front camera not available")
                return
            await self._stream_from(self.bc_front, websocket)
            return

        await websocket.close(code=1008, reason="Invalid path")

    async def _stream_from_gimbal(self, websocket):
        """Gimbal 전용: JPEG + Confidence 전송"""
        bc = self.bc_gimbal
        
        # 초기 프레임 전송
        if bc.latest:
            await self._safe_send(bc, websocket, bc.latest)
            conf_msg = json.dumps({"confidence": bc.latest_confidence})
            try:
                await asyncio.wait_for(websocket.send(conf_msg), timeout=SEND_TIMEOUT_SEC)
            except Exception:
                pass

        while not bc._stop.is_set():
            bc._frame_event.clear()
            await bc._frame_event.wait()
            if bc._stop.is_set():
                break
            if not bc.latest:
                continue
            
            # JPEG 전송 (binary)
            await self._safe_send(bc, websocket, bc.latest)
            
            # Confidence 전송 (text)
            conf_msg = json.dumps({"confidence": bc.latest_confidence})
            try:
                await asyncio.wait_for(websocket.send(conf_msg), timeout=SEND_TIMEOUT_SEC)
            except Exception:
                pass

    async def _stream_from(self, bc, websocket):
        """Front 카메라: JPEG만 전송 (기존 유지)"""
        if bc.latest:
            await self._safe_send(bc, websocket, bc.latest)

        while not bc._stop.is_set():
            bc._frame_event.clear()
            await bc._frame_event.wait()
            if bc._stop.is_set():
                break
            if not bc.latest:
                continue
            await self._safe_send(bc, websocket, bc.latest)

    async def _safe_send(self, bc, websocket, payload: bytes):
        try:
            await asyncio.wait_for(websocket.send(payload), timeout=SEND_TIMEOUT_SEC)
        except asyncio.TimeoutError:
            bc.stats.dropped_send += 1
        except Exception:
            bc.stats.send_fail += 1

async def main():
    # 1) IMX477 드론 탐지 카메라
    bc_gimbal = DroneDetectionBroadcaster("gimbal", TARGET_FPS_GIMBAL)
    bc_gimbal.start_camera()
    bc_gimbal.start_fifo_sender()
    cap_task_g = asyncio.create_task(bc_gimbal.capture_loop())

    # 2) USB 카메라
    bc_front = None
    cap_task_f = None
    try:
        bc_front = JpegBroadcaster("front", PIPELINE_FRONT, TARGET_FPS_FRONT)
        bc_front.start_gst()
        cap_task_f = asyncio.create_task(bc_front.capture_loop())
        print("[front] started OK")
    except Exception as e:
        bc_front = None
        print("[front] start FAILED:", e)

    router = RouterServer(bc_gimbal, bc_front)

    loop = asyncio.get_running_loop()
    def stop_all():
        bc_gimbal.request_stop()
        if bc_front:
            bc_front.request_stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop_all)

    print(f"\n✅ WebSocket server running on ws://{HOST}:{PORT}")
    print(f"   Gimbal (drone detection): ws://{HOST}:{PORT}{PATH_GIMBAL}")
    print(f"   Front (USB camera): ws://{HOST}:{PORT}{PATH_FRONT}\n")

    try:
        async with websockets.serve(
            router.ws_handler,
            HOST,
            PORT,
            max_size=None,
            ping_interval=10,
            ping_timeout=10,
        ):
            await bc_gimbal._stop.wait()
    finally:
        for t in (cap_task_g, cap_task_f):
            if t is None:
                continue
            t.cancel()
            with contextlib.suppress(Exception):
                await t

        bc_gimbal.stop_fifo_sender()
        bc_gimbal.stop_camera()
        if bc_front:
            bc_front.stop_gst()

        print("\n=== stats ===")
        print(f"[gimbal] frames: {bc_gimbal.stats.frames}, "
              f"dropped_send: {bc_gimbal.stats.dropped_send}, "
              f"send_fail: {bc_gimbal.stats.send_fail}, "
              f"sample_fail: {bc_gimbal.stats.sample_fail}")
        if bc_front:
            print(f"[front ] frames: {bc_front.stats.frames}, "
                  f"dropped_send: {bc_front.stats.dropped_send}, "
                  f"send_fail: {bc_front.stats.send_fail}, "
                  f"sample_fail: {bc_front.stats.sample_fail}")

if __name__ == "__main__":
    asyncio.run(main())