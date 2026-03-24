import os
import random
import shutil

# กำหนด path
input_dir = './imagedataset/wait_data_for_split'  # โฟลเดอร์ต้นทาง
target_base_dir = './imagedataset/splited_dataset'  # โฟลเดอร์ปลายทาง

# อัตราส่วนแบ่ง
split_ratios = {
    'train': 0.80,
    'valid': 0.20,
}

# สุ่มแบบเดิมทุกครั้ง
random.seed(42)

# เลือกเฉพาะไฟล์ภาพ
image_exts = ['.jpg', '.jpeg', '.png']
all_images = [
    f for f in os.listdir(input_dir)
    if os.path.splitext(f)[1].lower() in image_exts
]
random.shuffle(all_images)

# คำนวณการแบ่ง
total = len(all_images)
train_end = int(total * split_ratios['train'])

split_files = {
    'train': all_images[:train_end],
    'valid': all_images[train_end:]
}

# ดำเนินการย้ายไฟล์
for split, files in split_files.items():
    image_dir = os.path.join(target_base_dir, 'images', split)
    label_dir = os.path.join(target_base_dir, 'labels', split)
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    for img_file in files:
        base_name, _ = os.path.splitext(img_file)

        # ย้ายไฟล์ภาพ
        src_img = os.path.join(input_dir, img_file)
        dst_img = os.path.join(image_dir, img_file)
        shutil.copy2(src_img, dst_img)

        # ย้าย label ถ้ามี
        txt_file = base_name + '.txt'
        src_txt = os.path.join(input_dir, txt_file)
        dst_txt = os.path.join(label_dir, txt_file)

        if os.path.exists(src_txt):
            shutil.copy2(src_txt, dst_txt)

print("✅ แบ่ง dataset พร้อม label แล้ว:")
for k, v in split_files.items():
    print(f"{k}: {len(v)} รูป")
