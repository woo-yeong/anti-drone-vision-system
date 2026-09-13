import os
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
import cv2
from ultralytics import YOLO
import time
import torch
import socket
import pickle
import struct
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from datetime import datetime

print("="*50)
print("Drone Tracking System - TensorRT + DeepSORT (TCP)")
print("="*50)

print("\nLoading TensorRT model...")
model = YOLO('/home/wooyeong/PBL/yolo/best.engine')
print("TensorRT model loaded!")

# 클래스 0만 사용 (target)
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

# ID 관리 클래스
class IDManager:
    def __init__(self):
        self.next_id = 1
        self.active_ids = set()
    
    def get_id(self, deepsort_id, deepsort_to_our_id):
        """DeepSORT ID를 우리 ID로 매핑"""
        if deepsort_id in deepsort_to_our_id:
            return deepsort_to_our_id[deepsort_id]
        
        # 새 ID 할당
        new_id = self.next_id
        self.next_id += 1
        self.active_ids.add(new_id)
        deepsort_to_our_id[deepsort_id] = new_id
        return new_id
    
    def remove_id(self, our_id):
        """ID 제거"""
        if our_id in self.active_ids:
            self.active_ids.remove(our_id)
        
        # 모든 ID가 사라지면 리셋!
        if len(self.active_ids) == 0:
            self.next_id = 1
            print("[ID Manager] All drones lost - resetting IDs to 1")

# 확장 메모리
class ExtendedTracker:
    def __init__(self, memory_time=5.0):
        self.memory_time = memory_time
        self.lost_tracks = {}
    
    def update_lost(self, track_id, bbox, velocity, timestamp):
        """사라진 드론 정보 저장"""
        self.lost_tracks[track_id] = {
            'last_pos': np.array([(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2]),
            'last_vel': velocity,
            'last_seen': timestamp,
            'bbox_size': np.array([bbox[2]-bbox[0], bbox[3]-bbox[1]])
        }
    
    def remove_track(self, track_id):
        """특정 트랙 제거"""
        if track_id in self.lost_tracks:
            del self.lost_tracks[track_id]

# 초기화
id_manager = IDManager()
deepsort_to_our_id = {}
ext_tracker = ExtendedTracker(memory_time=10.0)

# 이전 프레임 추적 정보
prev_tracks = {}

# TCP 소켓 설정
LAPTOP_IP = "172.20.10.2"
PORT = 9999

print(f"\nConnecting to {LAPTOP_IP}:{PORT}...")

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 262144)

try:
    sock.connect((LAPTOP_IP, PORT))
    print("Connected successfully!")
except Exception as e:
    print(f"Connection failed: {e}")
    exit()

# GStreamer 파이프라인
pipeline = (
        "nvarguscamerasrc sensor-id=0 "
        "wbmode=1 "
        "saturation=0.5 "
        "exposurecompensation=0.5 "
        "! "
        "video/x-raw(memory:NVMM), width=1920, height=1080, framerate=30/1 ! "
        "nvvidconv ! "
        "video/x-raw, width=1920, height=1080, format=BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! "
        "appsink drop=true sync=false"
    )

print(f"\nOpening camera...")
cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("카메라 열기 실패!")
    sock.close()
    exit()

print("Camera opened successfully!")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Resolution: {width}x{height}")
print(f"Device: TensorRT (GPU) + DeepSORT")
print(f"Color correction: RGB*1.2, Hue-16")
print(f"YOLO inference: Every 2 frames")
print("Press Ctrl+C to quit\n")

frame_count = 0
prev_time = time.time()
fps_list = []

# 2프레임마다 YOLO 실행
YOLO_EVERY_N_FRAMES = 2

# 이전 탐지 결과 저장
last_detected_bboxes = set()
last_detection_confidences = {}

try:
    while cap.isOpened():
        success, frame = cap.read()
        
        if not success:
            print("Failed to read frame")
            break
        
        # ===== 색상 보정 적용 =====
        frame = color_correction(frame)
        
        frame_count += 1
        current_time = time.time()
        timestamp = datetime.now()
        fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
        prev_time = current_time
        fps_list.append(fps)
        avg_fps = sum(fps_list[-30:]) / len(fps_list[-30:])
        
        # 2프레임마다 YOLO 탐지
        if frame_count % YOLO_EVERY_N_FRAMES == 1:
            # YOLO 탐지
            results = model.predict(
                frame, 
                conf=0.4,
                iou=0.7,
                half=True,
                verbose=False,
                device=0  # GPU 명시
            )
            
            # 현재 프레임에서 실제 검출된 bbox 목록
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
            
            # 저장 (다음 프레임에서 사용)
            last_detected_bboxes = detected_bboxes
            last_detection_confidences = detection_confidences
        else:
            # 이전 탐지 결과 재사용
            detected_bboxes = last_detected_bboxes
            detection_confidences = last_detection_confidences
        
        # DeepSORT 포맷으로 변환
        detections_for_tracker = []
        for bbox_key in detected_bboxes:
            x1, y1, x2, y2 = bbox_key
            conf = detection_confidences[bbox_key]
            detections_for_tracker.append(([x1, y1, x2-x1, y2-y1], conf, 'target'))
        
        # DeepSORT 업데이트 (매 프레임)
        tracks = tracker.update_tracks(detections_for_tracker, frame=frame)
        
        # 현재 프레임의 활성 우리 ID들
        current_our_ids = set()
        
        detections = []
        detection_count = 0
        
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            deepsort_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            
            # 현재 프레임에서 실제 검출된 것인지 확인
            is_detected_now = False
            matched_conf = 0.0
            
            for bbox_key in detected_bboxes:
                bx1, by1, bx2, by2 = bbox_key
                if abs(x1-bx1) < 20 and abs(y1-by1) < 20 and abs(x2-bx2) < 20 and abs(y2-by2) < 20:
                    is_detected_now = True
                    matched_conf = detection_confidences[bbox_key]
                    break
            
            # 실제 검출된 것만 처리
            if not is_detected_now:
                continue
            
            # 우리 ID로 변환
            our_id = id_manager.get_id(deepsort_id, deepsort_to_our_id)
            current_our_ids.add(our_id)
            
            # 확장 메모리에서 제거 (다시 보임)
            ext_tracker.remove_track(our_id)
            
            # 속도 계산
            velocity = np.array([0.0, 0.0])
            if our_id in prev_tracks:
                prev_pos = prev_tracks[our_id]['pos']
                current_pos = np.array([(x1+x2)/2, (y1+y2)/2])
                dt = (timestamp - prev_tracks[our_id]['timestamp']).total_seconds()
                if dt > 0:
                    velocity = (current_pos - prev_pos) / dt
            
            # 현재 위치 저장
            prev_tracks[our_id] = {
                'pos': np.array([(x1+x2)/2, (y1+y2)/2]),
                'bbox': [x1, y1, x2, y2],
                'vel': velocity,
                'timestamp': timestamp
            }
            
            detection_count += 1
            
            dx, dy = (x1 + x2) // 2, (y1 + y2) // 2
            err_x = dx - (width // 2)
            err_y = dy - (height // 2)
            
            # 바운딩 박스 그리기
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (dx, dy), 5, (0, 0, 255), -1)
            
            # ID와 Confidence 표시
            cv2.putText(frame, f"ID:{our_id} ({matched_conf:.2f})", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            info_text = f"X:{err_x:4d} Y:{err_y:4d}"
            cv2.putText(frame, info_text, (x1, y2 + 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            detections.append({
                'id': our_id,
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'err_x': err_x, 'err_y': err_y,
                'confidence': matched_conf
            })
        
        # 사라진 드론 처리
        lost_our_ids = set(prev_tracks.keys()) - current_our_ids
        for our_id in lost_our_ids:
            if our_id in prev_tracks:
                info = prev_tracks[our_id]
                ext_tracker.update_lost(our_id, info['bbox'], info['vel'], info['timestamp'])
        
        # 확장 메모리에서 만료된 드론 확인 및 ID 회수
        expired_our_ids = []
        for our_id in list(ext_tracker.lost_tracks.keys()):
            info = ext_tracker.lost_tracks[our_id]
            dt = (timestamp - info['last_seen']).total_seconds()
            
            if dt > ext_tracker.memory_time:
                expired_our_ids.append(our_id)
                del ext_tracker.lost_tracks[our_id]
                if our_id in prev_tracks:
                    del prev_tracks[our_id]
        
        # ID 제거 및 리셋 체크
        for our_id in expired_our_ids:
            id_manager.remove_id(our_id)
            
            # DeepSORT 매핑에서도 제거
            for ds_id, mapped_id in list(deepsort_to_our_id.items()):
                if mapped_id == our_id:
                    del deepsort_to_our_id[ds_id]
                    break
        
        # 중심점 표시
        cv2.circle(frame, (width // 2, height // 2), 5, (255, 0, 0), -1)
        
        # FPS 및 정보 표시
        yolo_status = "YOLO" if frame_count % YOLO_EVERY_N_FRAMES == 1 else "Skip"
        cv2.putText(frame, f"FPS: {avg_fps:.1f} ({yolo_status})", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        status_text = f"Active: {detection_count} | Memory: {len(ext_tracker.lost_tracks)}"
        cv2.putText(frame, status_text, (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 프레임 인코딩
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
        _, encoded = cv2.imencode('.jpg', frame, encode_param)
        
        # 데이터 패킹
        data = pickle.dumps({
            'frame': encoded,
            'detections': detections,
            'fps': avg_fps,
            'frame_count': frame_count,
            'detection_count': detection_count,
            'memory_count': len(ext_tracker.lost_tracks)
        })
        
        # TCP로 전송
        size = len(data)
        try:
            sock.sendall(struct.pack('>I', size))
            sock.sendall(data)
        except Exception as e:
            print(f"Send error: {e}")
            break
        
        if frame_count % 30 == 0:
            print(f"Frames: {frame_count}, FPS: {avg_fps:.1f}, Size: {size//1024}KB, "
                  f"Active: {detection_count}, Memory: {len(ext_tracker.lost_tracks)}")

except KeyboardInterrupt:
    print("\n\nInterrupted by user")
except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()
finally:
    cap.release()
    sock.close()

print(f"\nTotal frames: {frame_count}")
if fps_list:
    print(f"Average FPS: {sum(fps_list)/len(fps_list):.1f}")