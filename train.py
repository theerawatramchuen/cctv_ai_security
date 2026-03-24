from ultralytics import YOLO

# โหลดโมเดล YOLOv11 แบบ OBB (Oriented Bounding Box)
model = YOLO ("yolo11s.pt")  # load a pretrained model (recommended for training) #(r"C:\Users\RYZEN\cctv_mon\project_yolov11_obb\runs\detect\train2\weights\best.pt") #
model.train(
    data="./data.yaml",
    epochs=200,
    imgsz=(640, 480),  # กว้าง x สูง ตามต้นฉบับ
    batch=8,
    workers=0,  # สำคัญ!
    cache=False, # โหลดรูปทั้งหมดไปยัง Memory ล่วงหน้าไหม ถ้าไม่คือ โหลดตามทีละ batch กำหนดไว้
    device=0,
    overlap_mask=False  # ใช้ Mask IoU loss สำหรับ OBB [เพิ่ม]
)