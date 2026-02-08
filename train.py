from ultralytics import YOLO

# 모델은 yolo 폴더 안의 것을 사용
model = YOLO(r'C:\Users\서우영\PBL\yolo\yolo11n.pt') 

# 데이터셋 설정 파일 위치 지정
model.train(
    data=r'C:\Users\서우영\PBL\datasets\data.yaml',
    epochs=50,
    imgsz=640,
    name='target_model'
)