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
print("Drone Detection System - TensorRT Mode (TCP)")
print("="*50)

print("\nLoading TensorRT model...")
model = YOLO('/home/wooyeong/PBL/yolo/best.engine')
print("TensorRT model loaded!")

# TCP 소켓 설정
LAPTOP_IP = "192.168.0.216"
PORT = 9999

print(f"\nConnecting to {LAPTOP_IP}:{PORT}...")

# TCP 소켓 생성 및 연결
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 262144)  # 256KB 버퍼

try:
    sock.connect((LAPTOP_IP, PORT))
    print("Connected successfully!")
except Exception as e:
    print(f"Connection failed: {e}")
    exit()

# GStreamer 파이프라인
pipeline = (
    "nvarguscamerasrc ! "
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
print(f"Device: TensorRT (GPU)")
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
        
        # YOLO 탐지 (TensorRT)
        results = model.predict(
            frame, 
            conf=0.5,
            verbose=False
        )
        
        detected = False
        detections = []
        detection_count = 0
        
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
        
        # FPS 표시 (TensorRT)
        cv2.putText(frame, f"FPS: {avg_fps:.1f} (TensorRT)", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # 현재 프레임의 탐지 개수
        status_text = f"Detections: {detection_count}"
        status_color = (0, 255, 0) if detected else (255, 255, 255)
        cv2.putText(frame, status_text, (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        # 프레임 인코딩
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
        
        # TCP로 전송 (크기 먼저, 그 다음 데이터)
        size = len(data)
        try:
            # 4바이트로 크기 전송
            sock.sendall(struct.pack('>I', size))
            # 실제 데이터 전송
            sock.sendall(data)
        except Exception as e:
            print(f"Send error: {e}")
            break
        
        if frame_count % 30 == 0:
            print(f"Frames: {frame_count}, FPS: {avg_fps:.1f}, Size: {size//1024}KB, Detections: {detection_count}")

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