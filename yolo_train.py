from ultralytics import YOLO

model = YOLO("yolov8n.pt")
results = model.train(data="data.yaml", epochs=300, imgsz=640, batch=64, device='cuda')
