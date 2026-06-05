# 🚗 License Plate Detection and Blurring using OpenCV

## 📌 Project Overview

This project implements an Automatic License Plate Detection and Privacy Protection System using OpenCV Haar Cascade Classifiers.

The system detects vehicle license plates from an image and automatically applies a blur effect to the detected region, helping protect sensitive information and improve privacy.

---

## 🎯 Objective

- Detect vehicle license plates from images.
- Apply blurring to detected license plates.
- Demonstrate image processing using OpenCV.
- Execute the entire workflow within JupyterLab.

---

## 🛠 Technologies Used

| Technology | Purpose |
|------------|----------|
| Python | Programming Language |
| OpenCV | Image Processing & Detection |
| Matplotlib | Image Visualization |
| JupyterLab | Development Environment |

---

## 📂 Project Structure

```text
License-Plate-Detection/
│
├── DATA/
│   ├── car_plate.jpg
│   └── haarcascades/
│       └── haarcascade_russian_plate_number.xml
│
├── License_Plate_Detection.ipynb
├── README.md
└── requirements.txt
```

---

## 📋 Requirements

Install the required libraries before running the notebook.

```bash
pip install opencv-python matplotlib
```
## ⚙️ Working Procedure

### Step 1
Load the input vehicle image using OpenCV.

### Step 2
Load the Haar Cascade XML classifier for license plate detection.

### Step 3
Convert the image into grayscale format.

### Step 4
Detect the license plate region using the cascade classifier.

### Step 5
Extract the detected license plate area.

### Step 6
Apply Median Blur to the detected plate region.

### Step 7
Replace the original plate region with the blurred output.

### Step 8
Display the final privacy-protected image.

---

## ▶️ Implementation Code

```python
import cv2
import matplotlib.pyplot as plt

def display(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10,6))
    plt.imshow(img_rgb)
    plt.axis('off')

img = cv2.imread("DATA/car_plate.jpg")
plate_cascade = cv2.CascadeClassifier(
    "DATA/haarcascades/haarcascade_russian_plate_number.xml"
)

def detect_and_blur_plate(img):
    img_copy = img.copy()
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)

    plates = plate_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4
    )

    for (x, y, w, h) in plates:
        roi = img_copy[y:y+h, x:x+w]
        blurred_roi = cv2.medianBlur(roi, 15)
        img_copy[y:y+h, x:x+w] = blurred_roi

    return img_copy

result = detect_and_blur_plate(img)
display(result)
```

---



---

## 📊 Output

The system successfully:

- Detects vehicle license plates.
- Applies blur to the detected region.
- Preserves the remaining image details.
- Enhances privacy protection.

---

## 📸 Result

### Original Image

<img width="716" height="362" alt="image" src="https://github.com/user-attachments/assets/7b1cc3e5-d199-4e33-a4b2-f2e0601d076e" />



### Output Image

<img width="647" height="357" alt="image" src="https://github.com/user-attachments/assets/de70261a-3840-40be-b6e0-ac98f1f9583e" />



---

## 🚀 Future Enhancements

- Automatic Number Plate Recognition (ANPR)
- OCR Integration using Tesseract
- Real-time Video Processing
- Vehicle Tracking System
- Deep Learning Based Plate Detection

---

## 👨‍💻 Author

**Naveen Kumar E**

Department of Artificial Intelligence and Data Science

Project: License Plate Detection and Blurring using OpenCV
