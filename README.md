# 👁️‍🗨️ Real-Time Facial Detection & Recognition – Android + Web (Flask)

This repository combines two parts of a lightweight, real-time facial detection and recognition system:

- 📱 `opencv_camera/` – Android app that captures face data using OpenCV and communicates with the backend.
- 💻 `python_opencv/` – Python Flask server that handles facial recognition, stores facial encodings, and performs authentication.

> 🚫 No deep learning, no training overhead – just efficient facial data encoding and `.pkl`-based user recognition!

---

## 📁 Repository Structure

```
opencv-face-recognition/
├── opencv_camera/ # Android app using OpenCV Camera
└── python_opencv/ # Flask backend for face detection & recognition
```


---

## 📱 Android App – `opencv_camera/`

An Android app that uses the OpenCV Camera module for real-time face capture.

### 🔧 Features

- Real-time face detection using OpenCV  
- Capture and send facial data to the Flask backend  
- Simple interface for collecting face data  

### 📦 Dependencies

- OpenCV Android SDK  
- OkHttp – for sending captured facial data to the backend  
- Camera permission handling  

---

## 💻 Python Flask Backend – `python_opencv/`

This is the heart of the recognition system – detects and recognizes users via webcam input.

### ✨ Features

- Face detection using Haar cascades (OpenCV)  
- Stores face encodings and names in `.pkl` file  
- Recognizes users in real-time  
- Simple Flask UI to test webcam functionality  

---

## 🧠 How It Works

- Facial data from Android or webcam is processed using OpenCV  
- Encodings are stored in `faces.pkl` using `pickle`  
- Flask app compares real-time input with stored encodings to identify users  

---

## 📦 Python Requirements

Install all dependencies:

```bash
pip install flask flask-cors scikit-learn pillow opencv-python-headless numpy
```

🚀 Getting Started

🖥️ For Flask Backend

```bash
cd python_opencv
pip install flask flask-cors scikit-learn pillow opencv-python-headless numpy
python test.py
```

📲 For Android App

Open ```opencv_camera``` in Android Studio

Connect your device Not emulator

Run the app and allow necessary camera permissions

🔁 Demo Flow

User opens the Android app or web → captures face

Face encoding is sent to Flask backend and saved

User is later recognized via webcam on the web and Android

---

📽️ Demo Video


🙌 Acknowledgements

Thanks to:

- OpenCV Community
- Flask Docs
- Scikit-learn for KNN
- Pythonistas everywhere

