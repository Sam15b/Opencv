#  👁️‍🗨️ Real-Time Facial Detection & Recognition – Android + Web (Flask)

This repository combines two parts of a lightweight, real-time facial detection and recognition system:

•📱 ```opencv_camera```/ – Android app that captures face data using OpenCV and communicates with the backend.

•💻 ```python_opencv```/ – Python Flask server that handles facial recognition, stores facial encodings, and performs authentication.

No deep learning, no training overhead – just efficient facial data encoding and .pkl-based user recognition!

📁 Repository Structure

```plaintext
├── opencv_camera/         # Android app using OpenCV Camera
└── python_opencv/         # Flask backend for face detection & recognition
```

📱 Android App – ```opencv_camera```/
An Android app that uses the OpenCV Camera module for real-time face capture.

🔧 Features

•Real-time face detection using OpenCV

•Capture and send facial data to the Flask backend

•Simple interface for collecting face data

📦 Dependencies

• OpenCV Android SDK

• OkHttp – for sending captured facial data to the backend

• Camera permission handling
 
 💻 Python Flask Backend – ```python_opencv/````

This is the heart of the recognition system – detects and recognizes users via webcam input.

✨ Features

• Face detection using Haar cascades (OpenCV)

• Stores face encodings and names in .pkl file

•Recognizes users in real-time

Simple Flask UI to test webcam functionality

🧠 How It Works

• Facial data from Android or webcam is processed using OpenCV

• Encodings are stored in trained_faces.pkl using pickle

• Flask app compares real-time input with stored encodings to identify users

📦 Python Requirements

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

📱 For Android App

1) Open ```opencv_camera``` in Android Studio.

2) Connect your device or emulator.

3) Run the app and allow necessary camera permissions.

🧪 Demo Flow

1) User opens the Android app And Web → captures face.

2) Face encoding is sent to Flask backend and saved.

3) User is later recognized via webcam on the web and Android Section.

📽️ Demo Video



🙌 Acknowledgements

Thanks to:

1) OpenCV Community

2) Flask Docs

3) Scikit-learn for KNN

4) Pythonistas everywhere
