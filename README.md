# 🏠 Smart Home AI Ecosystem

<div align="center">
  <h3>Powered by Artificial Intelligence & Machine Learning</h3>
  <p>A modernized, glassmorphism-themed locally hosted web application integrating cutting edge machine learning elements for a truly autonomous smart home experience.</p>
</div>

---

## 🚀 Features

This project utilizes 4 core AI pipelines to simulate and control a futuristic smart home environment:

#### 1. 🌀 Fan Speed Prediction
* **Technology:** Real-time Decision Tree Architecture (Scikit-Learn).
* **How it works:** Input the ambient room temperature, and the AI calculates and outputs the mathematically optimal fan spin speed based on thousands of data points.

#### 2. 💡 Emotion Mood Lighting
* **Technology:** Computer Vision (OpenCV) + Machine Learning (CNN/SVM Classifier).
* **How it works:** The system seamlessly connects to your webcam, scans your facial expression to deduce your current emotional state (Happy, Sad, Angry, Neutral, etc.), and instantly manipulates dynamic "healing" room lights that counteract negative emotions or amplify positive ones.

#### 3. 🚪 Face Recognition Security Hub
* **Technology:** OpenCV LBPH (Local Binary Patterns Histograms) Face Recognizer.
* **How it works:** Military-grade secure gatekeeper. Only authorized faces registered by an administrator via the secure panel can unlock the main door. Attempts to bypass are scanned and securely denied.

#### 4. 🎤 Multilingual Voice Command Center
* **Technology:** Offline Voice Recognition (Vosk KaldiRecognizer).
* **How it works:** Full offline control of the house using spoken language. Features dual support for **English** and **Hindi**. Say commands like "turn on the fan" or "open the door" and the system will actively execute.

---

## 🎨 UI/UX Design

The entire user dashboard utilizes **Glassmorphism**, built purely in raw native Vanilla CSS and JavaScript, avoiding heavy frontend frameworks.
- Deep neon-accented dark mode.
- Context-aware background blur filters.
- Fluid micro-animations.
- Enterprise-aligned typography and responsive structural layouts.

---

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone <your-repository-url>
   cd smart-home-ai
   ```

2. **Install the dependencies:**
   Make sure you have Python installed. You can install all required AI libraries via:
   ```bash
   pip install -r requirements.txt
   ```

3. **Required Machine Learning Models:**
   **(Note: Make sure your `models/` directory has been correctly populated with the required pre-trained files natively, e.g., Vosk language extraction and Emotion classifiers).*

4. **Launch the Server:**
   ```bash
   python app.py
   ```
   *The server will initialize the Flask backend on `http://127.0.0.1:5001` or `localhost:5001`.*

---

## 🔐 Administrative Settings

The AI dashboard houses secure sub-menus for registering new identities (Face IDs and Voice Profiles).
- **Default System Admin Passcode:** `vansh123`
*(Ensure you change this passcode in `app.js` prior to hosting publicly).*

---

## 📄 License & Privacy

This project runs 100% locally. No voice audio, facial data, or biometric imagery leaves your device. Data is evaluated entirely on-device via locally hosted `Vosk` and `OpenCV` instances.
