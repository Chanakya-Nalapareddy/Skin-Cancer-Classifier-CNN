# Skin-Cancer-Classifier-CNN
A CNN-based skin cancer classification system built in Python to detect 9 types of skin cancer from ISIC dermoscopic images using deep learning and data augmentation.

# Skin Cancer Detection

This project focuses on the detection of skin cancer using deep learning models. It involves preprocessing, augmenting image data, and training various Convolutional Neural Network (CNN) architectures.

## Project Structure

The structure of the final project folder

```plaintext
.
└── project
    ├── README.md
    ├── data
    │   └── Skin_cancer_ISIC_The_International_Skin_Imaging_Collaboration
    │       ├── Test
    │       │   ├── actinic keratosis
    │       │   ├── basal cell carcinoma
    │       │   ├── dermatofibroma
    │       │   ├── melanoma
    │       │   ├── nevus
    │       │   ├── pigmented benign keratosis
    │       │   ├── seborrheic keratosis
    │       │   ├── squamous cell carcinoma
    │       │   └── vascular lesion
    │       ├── Train
    │       │   ├── actinic keratosis
    │       │   ├── basal cell carcinoma
    │       │   ├── dermatofibroma
    │       │   ├── melanoma
    │       │   ├── nevus
    │       │   ├── pigmented benign keratosis
    │       │   ├── seborrheic keratosis
    │       │   ├── squamous cell carcinoma
    │       │   └── vascular lesion
    │       └── processed
    │           ├── color
    │           │   ├── test
    │           │   │   ├── actinic keratosis
    │           │   │   ├── basal cell carcinoma
    │           │   │   ├── dermatofibroma
    │           │   │   ├── melanoma
    │           │   │   ├── nevus
    │           │   │   ├── pigmented benign keratosis
    │           │   │   ├── seborrheic keratosis
    │           │   │   ├── squamous cell carcinoma
    │           │   │   └── vascular lesion
    │           │   └── train
    │           │       ├── actinic keratosis
    │           │       ├── basal cell carcinoma
    │           │       ├── dermatofibroma
    │           │       ├── melanoma
    │           │       ├── nevus
    │           │       ├── pigmented benign keratosis
    │           │       ├── seborrheic keratosis
    │           │       ├── squamous cell carcinoma
    │           │       └── vascular lesion
    │           └── gray
    │               ├── test
    │               │   ├── actinic keratosis
    │               │   ├── basal cell carcinoma
    │               │   ├── dermatofibroma
    │               │   ├── melanoma
    │               │   ├── nevus
    │               │   ├── pigmented benign keratosis
    │               │   ├── seborrheic keratosis
    │               │   ├── squamous cell carcinoma
    │               │   └── vascular lesion
    │               └── train
    │                   ├── actinic keratosis
    │                   ├── basal cell carcinoma
    │                   ├── dermatofibroma
    │                   ├── melanoma
    │                   ├── nevus
    │                   ├── pigmented benign keratosis
    │                   ├── seborrheic keratosis
    │                   ├── squamous cell carcinoma
    │                   └── vascular lesion
    ├── framework
    │   ├── ConvolutionalLayer.py
    │   ├── CrossEntropy.py
    │   ├── FlatteningLayer.py
    │   ├── FullyConnectedLayer.py
    │   ├── InputLayer.py
    │   ├── Layer.py
    │   ├── MaxPoolLayer.py
    │   ├── ReLULayer.py
    │   ├── SoftmaxLayer.py
    │   ├── __init__.py
    ├── model
    │   ├── cnn_color.py
    │   ├── cnn_color_kernels.py
    │   ├── cnn_color_layers_kernels.py
    │   ├── cnn_gray.py
    │   ├── cnn_gray_kernels.py
    │   └── cnn_gray_layers_kernels.py
    └── preprocessing
        ├── augmentation.py
        └── preprocessing.py
```

## Running the Project

### Data Preprocessing and Augmentation

There are two ways to initiate working with the dataset. You can either start with preprocessing, where a subset of images is randomly selected from the original dataset, or begin with augmentation, where the dataset is enhanced to address class imbalance. You can choose either method to proceed based on your requirements.

Navigate to the preprocessing directory and execute:

```python
python preprocessing.py
python augmentation.py
```
- **`preprocessing.py`**:
  - Randomly selects 60 images per class for training and 20 images per class for testing from the original dataset.
  - Resizes images to a uniform size.
  - Saves processed images in both color and grayscale formats.

- **`augmentation.py`**:
  - Applies augmentation techniques (rotation, flipping, scaling) to the original dataset.
  - Balances class distribution.
  - Enhances diversity of the training data.

## Model Training

Navigate to the **model** directory and execute the CNN architectures:

Each Python script corresponds to a distinct CNN architecture tailored for color and grayscale image datasets.

```python
python cnn_color.py
python cnn_color_kernels.py
python cnn_color_layers_kernels.py
python cnn_gray.py
python cnn_gray_kernels.py
python cnn_gray_layers_kernels.py
```

<br>

---

<br>

Developed as part of [**CS 615 - Deep Learning / Skin Cancer Detection**], [03/20/2025]

