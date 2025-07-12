import os
import random
import shutil
from PIL import Image

base_dir = "../data/Skin_cancer_ISIC_The_International_Skin_Imaging_Collaboration"
original_train_dir = os.path.join(base_dir, "Train")
original_test_dir = os.path.join(base_dir, "Test")
processed_dir = os.path.join(base_dir, "processed")
color_dir = os.path.join(processed_dir, "color")
gray_dir = os.path.join(processed_dir, "gray")
color_train_dir = os.path.join(color_dir, "train")
color_test_dir = os.path.join(color_dir, "test")
gray_train_dir = os.path.join(gray_dir, "train")
gray_test_dir = os.path.join(gray_dir, "test")

classes = [
    "actinic keratosis", "basal cell carcinoma", "dermatofibroma", "melanoma",
    "nevus", "pigmented benign keratosis", "seborrheic keratosis",
    "squamous cell carcinoma", "vascular lesion"
]

if os.path.exists(processed_dir):
    shutil.rmtree(processed_dir)
    print(f"Removed existing '{processed_dir}' folder.")

os.makedirs(color_train_dir, exist_ok=True)
os.makedirs(color_test_dir, exist_ok=True)
os.makedirs(gray_train_dir, exist_ok=True)
os.makedirs(gray_test_dir, exist_ok=True)
temp_dir = os.path.join(base_dir, "Temporary")
os.makedirs(temp_dir, exist_ok=True)

def preprocess_color_image(input_path, output_path, size=(32, 32)):
    try:
        img = Image.open(input_path).convert("RGB")
        img = img.resize(size)
        img.save(output_path, "JPEG")
        verify_img = Image.open(output_path)
        if verify_img.size != size:
            print(f"Warning: {output_path} saved with size {verify_img.size}, not {size}")
    except Exception as e:
        print(f"Error processing color {input_path}: {e}")

def preprocess_gray_image(input_path, output_path, size=(32, 32)):
    try:
        img = Image.open(input_path).convert("L")
        img = img.resize(size)
        img.save(output_path, "JPEG")
        verify_img = Image.open(output_path)
        if verify_img.size != size:
            print(f"Warning: {output_path} saved with size {verify_img.size}, not {size}")
    except Exception as e:
        print(f"Error processing gray {input_path}: {e}")

for class_name in classes:
    train_class_dir = os.path.join(original_train_dir, class_name)
    test_class_dir = os.path.join(original_test_dir, class_name)
    temp_class_dir = os.path.join(temp_dir, class_name)
    os.makedirs(temp_class_dir, exist_ok=True)
    
    for img in os.listdir(train_class_dir):
        input_path = os.path.join(train_class_dir, img)
        output_path = os.path.join(temp_class_dir, f"train_{img}")
        preprocess_color_image(input_path, output_path)
    for img in os.listdir(test_class_dir):
        input_path = os.path.join(test_class_dir, img)
        output_path = os.path.join(temp_class_dir, f"test_{img}")
        preprocess_color_image(input_path, output_path)

for class_name in classes:
    os.makedirs(os.path.join(color_train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(color_test_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(gray_train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(gray_test_dir, class_name), exist_ok=True)

for class_name in classes:
    temp_class_dir = os.path.join(temp_dir, class_name)
    color_train_class_dir = os.path.join(color_train_dir, class_name)
    color_test_class_dir = os.path.join(color_test_dir, class_name)
    gray_train_class_dir = os.path.join(gray_train_dir, class_name)
    gray_test_class_dir = os.path.join(gray_test_dir, class_name)
    
    images = os.listdir(temp_class_dir)
    random.shuffle(images)
    
    train_images = images[:70]
    test_images = images[70:80]
    
    for img in train_images:
        src = os.path.join(temp_class_dir, img)
        final_name = img.replace("train_", "").replace("test_", "")
        dst = os.path.join(color_train_class_dir, final_name)
        shutil.copy(src, dst)
    for img in test_images:
        src = os.path.join(temp_class_dir, img)
        final_name = img.replace("train_", "").replace("test_", "")
        dst = os.path.join(color_test_class_dir, final_name)
        shutil.copy(src, dst)

    for img in train_images:
        final_name = img.replace("train_", "").replace("test_", "")
        color_path = os.path.join(color_train_class_dir, final_name)
        gray_path = os.path.join(gray_train_class_dir, final_name)
        preprocess_gray_image(color_path, gray_path)
    for img in test_images:
        final_name = img.replace("train_", "").replace("test_", "")
        color_path = os.path.join(color_test_class_dir, final_name)
        gray_path = os.path.join(gray_test_class_dir, final_name)
        preprocess_gray_image(color_path, gray_path)

shutil.rmtree(temp_dir)

print("Redistribution complete: 60 train, 20 test per class in 'processed/color', 'processed/gray' folder.")

