from ultralytics import YOLO

def main():
    # Model oluştur
    model = YOLO("yolov8s.pt")

    # Eğitimi başlat
    model.train(
        data="data.yaml",
        epochs=50,
        imgsz=640,
        batch=8,
        workers=4,
        device=0,
        cache=False
    )

if __name__ == "__main__":
    main()