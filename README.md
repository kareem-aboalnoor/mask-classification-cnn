# Mask Classification Using CNN

This project is a simple deep learning model that classifies whether a person is **wearing a mask** or **not wearing a mask** using a CNN (MobileNetV2).  
The project includes real-time/video prediction using OpenCV.

---

## 🔥 Features
- Mask vs. No Mask classification
- Transfer Learning (MobileNetV2)
- Real-time detection using OpenCV
- Color-coded output:
  - Green = Mask
  - Red = No Mask
- Lightweight `.h5` model

---

## 📂 Project Structure
```
project/
│── main.py
│── train.py
│── requirements.txt
│── README.md
│
├── model/
│     └── mask_detector.h5
│
└── dataset/
      ├── mask/
      └── no_mask/
```

---

## ▶️ Run Real-Time / Video Detection
### Webcam:
```bash
python main.py
```

### Video:
```bash
python main.py --video test.mp4
```

---

## 🧠 Training the Model
Dataset format must be:
```
dataset/
   ├── mask/
   └── no_mask/
```

Train:
```bash
python train.py --dataset dataset --model model/mask_detector.h5 --epochs 10
```

---

## 📦 Install Requirements
```bash
pip install -r requirements.txt
```

---

## 🛠 Technologies Used
- TensorFlow / Keras
- MobileNetV2
- NumPy
- OpenCV

---

## 📝 Notes
- `model/mask_detector.h5` may not be uploaded due to GitHub size limits.
- You can upload it on Google Drive and link it in this README.

