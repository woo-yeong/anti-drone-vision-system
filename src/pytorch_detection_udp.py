import os
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
import cv2
from ultralytics import YOLO
import time
import torch
import socket
import pickle
import struct

print("="*50)
print("Drone Detection System - Streaming Mode")
print("="*50)
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("="*50)

print("\nLoading YOLO model...")
model = YOLO('/home/wooyeong/PBL/yolo/best.pt')
print("Model loaded!")

# UDP 소켓 설정
LAPTOP_IP = "172.20.10.2"  # 노트북 IP로 변경하세요!
PORT = 9999

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65536)

print(f"\nStreaming to {LAPTOP_IP}:{PORT}")

# GStreamer 파이프라인
pipeline = (
    "nvarguscamerasrc ! "
    "video/x-raw(memory:NVMM), width=640, height=480, framerate=30/1 ! "
    "nvvidconv ! "
    "video/x-raw, width=640, height=480, format=BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=BGR ! "
    "appsink drop=true sync=false"
)

print(f"\nOpening camera...")
cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("카메라 열기 실패!")
    exit()

print("Camera opened successfully!")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
use_gpu = torch.cuda.is_available()
device = 0 if use_gpu else 'cpu'

print(f"Resolution: {width}x{height}")
print(f"Device: {'GPU' if use_gpu else 'CPU'}")
print("Press Ctrl+C to quit\n")

frame_count = 0
prev_time = time.time()
fps_list = []

try:
    while cap.isOpened():
        success, frame = cap.read()
        
        if not success:
            print("Failed to read frame")
            break
        
        frame_count += 1
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
        prev_time = current_time
        fps_list.append(fps)
        avg_fps = sum(fps_list[-30:]) / len(fps_list[-30:])
        
        # YOLO 탐지
        results = model.predict(
            frame, 
            conf=0.5,
            show=False, 
            verbose=False,
            imgsz=416,  # 메모리 절약
            half=use_gpu,
            device=device
        )
        
        detected = False
        detections = []
        detection_count = 0  # 현재 프레임의 탐지 개수
        
        for r in results:
            for box in r.boxes:
                detected = True
                detection_count += 1
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                dx, dy = (x1 + x2) // 2, (y1 + y2) // 2
                err_x = dx - (width // 2)
                err_y = dy - (height // 2)
                conf = float(box.conf[0])
                
                # 바운딩 박스 그리기
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (dx, dy), 5, (0, 0, 255), -1)
                
                info_text = f"X:{err_x:4d} Y:{err_y:4d} C:{conf:.2f}"
                cv2.putText(frame, info_text, (x1, y2 + 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                
                detections.append({
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'err_x': err_x, 'err_y': err_y, 'conf': conf
                })
        
        # 중심점 표시
        cv2.circle(frame, (width // 2, height // 2), 5, (255, 0, 0), -1)
        
        # FPS 표시
        color = (0, 255, 0) if use_gpu else (255, 255, 0)
        device_text = "GPU" if use_gpu else "CPU"
        cv2.putText(frame, f"FPS: {avg_fps:.1f} ({device_text})", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # 현재 프레임의 탐지 개수만 표시
        status_text = f"Detections: {detection_count}"
        status_color = (0, 255, 0) if detected else (255, 255, 255)
        cv2.putText(frame, status_text, (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        # 프레임 인코딩 및 전송
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
        _, encoded = cv2.imencode('.jpg', frame, encode_param)
        
        # 데이터 패킹
        data = pickle.dumps({
            'frame': encoded,
            'detections': detections,
            'fps': avg_fps,
            'frame_count': frame_count,
            'detection_count': detection_count
        })
        
        # UDP로 전송 (패킷 크기 제한 64KB)
        size = len(data)
        if size < 65000:
            try:
                sock.sendto(data, (LAPTOP_IP, PORT))
            except Exception as e:
                print(f"Send error: {e}")
        else:
            print(f"Frame too large: {size} bytes")
        
        if frame_count % 30 == 0:
            print(f"Frames: {frame_count}, FPS: {avg_fps:.1f}, Detections: {detection_count}")

except KeyboardInterrupt:
    print("\n\nInterrupted by user")
except Exception as e:
    print(f"\nError: {e}")
finally:
    cap.release()
    sock.close()

print(f"\nTotal frames: {frame_count}")
if fps_list:
    print(f"Average FPS: {sum(fps_list)/len(fps_list):.1f}")