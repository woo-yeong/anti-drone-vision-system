# Edge-AI Drone Detection & Tracking

Edge-AI 기반 안티드론 UGV 프로젝트에서 사용한 Jetson 기반 드론 탐지·추적 소프트웨어 코드입니다.

## 주요 기능

- YOLO 기반 드론 객체 탐지
- TensorRT 및 PyTorch 기반 추론 실행
- DeepSORT·ByteTrack 기반 객체 추적
- GStreamer 기반 CSI/USB 카메라 입력 처리
- WebSocket·TCP·UDP 기반 영상 및 탐지 결과 전송
- 탐지 객체 중심 좌표를 FIFO로 전달해 후단 제어 시스템과 연동
- Node.js 기반 WebSocket 중계 서버

## 디렉터리 구성

```text
.
├── src/
│   ├── tensorrt_bytetrack_websocket.py
│   ├── tensorrt_deepsort_websocket.py
│   ├── tensorrt_deepsort_tcp.py
│   ├── tensorrt_detection_tcp.py
│   └── pytorch_detection_udp.py
├── training/
│   └── train_yolo11.py
├── configs/
│   └── botsort_lightweight.yaml
├── websocket_server/
│   ├── server.js
│   ├── package.json
│   └── package-lock.json
└── README.md
```

## 파일 설명

- `tensorrt_bytetrack_websocket.py` : TensorRT YOLO 추론과 ByteTrack 추적, WebSocket 영상 전송 및 좌표 출력
- `tensorrt_deepsort_websocket.py` : TensorRT YOLO 추론과 DeepSORT 추적, WebSocket 영상 전송 및 좌표 출력
- `tensorrt_deepsort_tcp.py` : TensorRT YOLO + DeepSORT 결과를 TCP로 전송하는 실험 코드
- `tensorrt_detection_tcp.py` : TensorRT 기반 YOLO 탐지 결과와 영상을 TCP로 전송하는 코드
- `pytorch_detection_udp.py` : PyTorch 기반 YOLO 탐지 결과와 영상을 UDP로 전송하는 비교 코드
- `train_yolo11.py` : YOLO11 객체 탐지 모델 학습 코드
- `botsort_lightweight.yaml` : ReID·CMC를 비활성화한 경량 BoT-SORT 설정
- `websocket_server/` : WebSocket 메시지 중계 서버

## 개발 환경

- NVIDIA Jetson Orin Nano
- Python
- OpenCV
- Ultralytics YOLO
- TensorRT
- GStreamer
- DeepSORT / ByteTrack
- WebSocket / TCP / UDP

> 모델 가중치와 장치별 경로·네트워크 설정은 실행 환경에 맞게 별도로 구성해야 합니다.
