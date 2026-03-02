import cv2
from ultralytics import YOLO

# ===========================
# 1️⃣ Modeli yükle
# ===========================
# Eğitim sonrası kaydettiğin best.pt modelinin yolunu yaz
MODEL_PATH = "C:/Users/ACER/Downloads/raw_data_v2-20260302T165127Z-1-001/raw_data_v2/runs/detect/train6/weights/best.pt"
model = YOLO(MODEL_PATH)

# ===========================
# 2️⃣ Realtime kamera
# ===========================
def realtime_camera_test():
    cap = cv2.VideoCapture(0)  # 0 = default laptop kamerası

    print("Realtime test başladı. Kameraya pamuk göster. Çıkmak için 'q' tuşuna bas.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Kamera açılamıyor!")
            break

        # FPS için frame boyutunu küçültebilirsin (opsiyonel)
        frame = cv2.resize(frame, (640, 640))

        # YOLO tahmini
        results = model(frame, conf=0.2, iou=0.4)

        # Tahmin kutucuklarını çiz
        annotated_frame = results[0].plot()

        # Görüntüyü göster
        cv2.imshow("Pamuk Realtime Detection", annotated_frame)

        # 'q' tuşuna basınca çık
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ===========================
# 3️⃣ Opsiyonel: Tek resim test
# ===========================
def single_image_test(image_path):
    results = model(image_path)
    results[0].show()  # Tahmin kutucuklarını göster
    # Dilersen sonucu kaydet
    # results[0].save("results_output.jpg")

# ===========================
# 4️⃣ Kullanıcı Seçimi
# ===========================
if __name__ == "__main__":
    print("1: Realtime kamera testi")
    print("2: Tek resim testi (telefon fotoğrafı)")
    choice = input("Seçiminiz (1/2): ")

    if choice == "1":
        realtime_camera_test()
    elif choice == "2":
        image_path = input("Resim yolunu girin (örn: pamuk.jpg): ")
        single_image_test(image_path)
    else:
        print("Geçersiz seçim!")