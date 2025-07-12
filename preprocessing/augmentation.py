import os
import random
import shutil
from PIL import Image, ImageEnhance
import numpy as np
from PIL import ImageFilter

try:
    import cv2

    OPENCV_AVAILABLE = True
except ImportError:
    print("OpenCV not installed. Shearing augmentation will be skipped.")
    OPENCV_AVAILABLE = False

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
    "actinic keratosis",
    "basal cell carcinoma",
    "dermatofibroma",
    "melanoma",
    "nevus",
    "pigmented benign keratosis",
    "seborrheic keratosis",
    "squamous cell carcinoma",
    "vascular lesion",
]

# Target number of images
TARGET_TRAIN_IMAGES = 60
TARGET_TEST_IMAGES = 20

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
            print(
                f"Warning: {output_path} saved with size {verify_img.size}, not {size}"
            )
    except Exception as e:
        print(f"Error processing color {input_path}: {e}")


def preprocess_gray_image(input_path, output_path, size=(32, 32)):
    try:
        img = Image.open(input_path).convert("L")
        img = img.resize(size)
        img.save(output_path, "JPEG")
        verify_img = Image.open(output_path)
        if verify_img.size != size:
            print(
                f"Warning: {output_path} saved with size {verify_img.size}, not {size}"
            )
    except Exception as e:
        print(f"Error processing gray {input_path}: {e}")


# Augmentation functions
def add_noise(image, noise_factor=0.05):
    img_array = np.array(image)
    noise = np.random.normal(loc=0, scale=noise_factor * 255, size=img_array.shape)
    noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_img)


def shear_image(image, shear_factor=0.2):
    if not OPENCV_AVAILABLE:
        return image
    img_array = np.array(image)
    rows, cols, ch = img_array.shape
    src_points = np.float32([[0, 0], [cols, 0], [0, rows]])
    dst_points = np.float32([[0, 0], [cols, 0], [shear_factor * cols, rows]])
    matrix = cv2.getAffineTransform(src_points, dst_points)
    sheared = cv2.warpAffine(img_array, matrix, (cols, rows))
    return Image.fromarray(sheared)


def zoom_image(image, zoom_factor=1.2):
    width, height = image.size
    new_width, new_height = int(width * zoom_factor), int(height * zoom_factor)
    img = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    left = (new_width - width) // 2
    top = (new_height - height) // 2
    return img.crop((left, top, left + width, top + height))


def augment_image(image):
    augmentations = []
    img = image.convert("RGB")

    # Original image
    augmentations.append(img)

    # Rotate 90, 180, 270 degrees
    for angle in [90, 180, 270]:
        augmentations.append(img.rotate(angle))

    # Horizontal flip
    augmentations.append(img.transpose(Image.FLIP_LEFT_RIGHT))

    # Vertical flip
    augmentations.append(img.transpose(Image.FLIP_TOP_BOTTOM))

    # Brightness adjustment
    enhancer = ImageEnhance.Brightness(img)
    augmentations.append(enhancer.enhance(0.8))  # Darker
    augmentations.append(enhancer.enhance(1.2))  # Brighter

    # Contrast adjustment
    enhancer = ImageEnhance.Contrast(img)
    augmentations.append(enhancer.enhance(0.8))  # Lower contrast
    augmentations.append(enhancer.enhance(1.2))  # Higher contrast

    # Add Gaussian noise
    augmentations.append(add_noise(img, noise_factor=0.03))
    augmentations.append(add_noise(img, noise_factor=0.05))

    # Zoom (slight zoom in and out)
    augmentations.append(zoom_image(img, zoom_factor=1.1))
    augmentations.append(zoom_image(img, zoom_factor=0.9))

    # Slight shear
    if OPENCV_AVAILABLE:
        augmentations.append(shear_image(img, shear_factor=0.1))
        augmentations.append(shear_image(img, shear_factor=-0.1))

    return augmentations


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

# Create final directories
for class_name in classes:
    os.makedirs(os.path.join(color_train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(color_test_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(gray_train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(gray_test_dir, class_name), exist_ok=True)

# Augment and distribute images
for class_name in classes:
    temp_class_dir = os.path.join(temp_dir, class_name)
    color_train_class_dir = os.path.join(color_train_dir, class_name)
    color_test_class_dir = os.path.join(color_test_dir, class_name)
    gray_train_class_dir = os.path.join(gray_train_dir, class_name)
    gray_test_class_dir = os.path.join(gray_test_dir, class_name)

    images = os.listdir(temp_class_dir)
    random.shuffle(images)

    train_images = [img for img in images if img.startswith("train_")]
    test_images = [img for img in images if img.startswith("test_")]

    train_count = 0
    train_idx = 0
    while train_count < TARGET_TRAIN_IMAGES:
        img_name = train_images[train_idx % len(train_images)]
        img_path = os.path.join(temp_class_dir, img_name)

        img = Image.open(img_path)
        augmented_images = augment_image(img)

        for aug_idx, aug_img in enumerate(augmented_images):
            if train_count >= TARGET_TRAIN_IMAGES:
                break

            final_name = (
                f"{train_count}_{class_name}_{img_name.split('.')[0]}_aug{aug_idx}.jpg"
            )
            aug_img.save(os.path.join(color_train_class_dir, final_name))
            train_count += 1

        train_idx += 1

    test_count = 0
    test_idx = 0
    while test_count < TARGET_TEST_IMAGES:
        img_name = test_images[test_idx % len(test_images)]
        img_path = os.path.join(temp_class_dir, img_name)

        img = Image.open(img_path)
        augmented_images = augment_image(img)

        for aug_idx, aug_img in enumerate(augmented_images):
            if test_count >= TARGET_TEST_IMAGES:
                break

            final_name = (
                f"{test_count}_{class_name}_{img_name.split('.')[0]}_aug{aug_idx}.jpg"
            )
            aug_img.save(os.path.join(color_test_class_dir, final_name))
            test_count += 1

        test_idx += 1

    # Create grayscale versions
    for img in os.listdir(color_train_class_dir):
        color_path = os.path.join(color_train_class_dir, img)
        gray_path = os.path.join(gray_train_class_dir, img)
        preprocess_gray_image(color_path, gray_path)

    for img in os.listdir(color_test_class_dir):
        color_path = os.path.join(color_test_class_dir, img)
        gray_path = os.path.join(gray_test_class_dir, img)
        preprocess_gray_image(color_path, gray_path)

shutil.rmtree(temp_dir)

print(
    f"Redistribution complete: {TARGET_TRAIN_IMAGES} train, {TARGET_TEST_IMAGES} test per class in 'processed/color', 'processed/gray' folder."
)
