from ultralytics import YOLO


model = YOLO("C:/Users/ACER/Downloads/raw_data_v2-20260302T165127Z-1-001/raw_data_v2/runs/detect/train6/weights/best.pt")


test_image = "C:/Users/ACER/Downloads/raw_data_v2-20260302T165127Z-1-001/raw_data_v2/images/test/DSC03644.JPG"


results = model(test_image)


results[0].show()   
