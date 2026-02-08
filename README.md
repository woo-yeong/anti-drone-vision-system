# IP 주소 확인 (코드 수정 필수)
ifconfig

# GUI 연결
PBL 디렉토리 이동 후 
python3 stream.py

Loading /home/wooyeong/PBL/yolo/best2.engine for TensorRT inference...
[02/08/2026-15:17:14] [TRT] [I] Loaded engine size: 8 MiB
[02/08/2026-15:17:14] [TRT] [W] Using an engine plan file across different models of devices is not recommended and is likely to affect performance or even cause errors.
[02/08/2026-15:17:14] [TRT] [I] [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +0, GPU +9, now: CPU 0, GPU 14 (MiB)

위의 로그가 떠야 정상

# GUI 연결 안될 때
# 카메라 연결 확인
ls /dev/viedo*
-> /dev/video1 부터 /dev/video3까지 뜸
아무것도 안 뜨면 재부팅

# 터미널 내 카메라 확인
nvgstcapture-1.0 
안되면 아래 명령 실행 후 다시 
sudo systemctl restart nvargus-daemon

# camera generated 관련 ERROR 뜰 때
sudo systemctl restart nvargus-daemon
혹은
sudo pkill nvargus-daemon
sudo systemctl start nvargus-daemon

그래도 안되면 재부팅



[gimbal] Opening FIFO (waiting for C program)...
[gimbal] FIFO error: [Errno 13] Permission denied: '/tmp/drone_coords'

-> sudo chmod 666 /tmp/drone_coords


# PBL
