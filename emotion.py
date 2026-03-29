"""
Feature 2: Emotion Detection → Healing Mood Lighting
Uses SVM + HOG features (scikit-learn) — works on all CPUs
Falls back to TensorFlow CNN if available
OpenCV Haar Cascade for face detection
"""

import numpy as np
import cv2
import os
import base64

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']


def extract_hog_features(image):
    """Extract HOG + grid stats + pixel features from a 48x48 grayscale image."""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.shape != (48, 48):
        image = cv2.resize(image, (48, 48))

    # 1. HOG features
    win_size = (48, 48)
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (8, 8)
    nbins = 9
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    hog_features = hog.compute(image).flatten()

    # 2. Grid statistics (4x4 grid)
    grid_stats = []
    h, w = image.shape
    gh, gw = h // 4, w // 4
    for i in range(4):
        for j in range(4):
            region = image[i*gh:(i+1)*gh, j*gw:(j+1)*gw]
            grid_stats.extend([
                np.mean(region),
                np.std(region),
                np.median(region),
            ])
    grid_stats = np.array(grid_stats)

    # 3. Downsampled pixels
    small = cv2.resize(image, (12, 12)).flatten().astype(np.float32)

    return np.concatenate([hog_features, grid_stats, small])


class EmotionDetector:
    def __init__(self):
        self.model = None
        self.model_type = None  # 'svm' or 'cnn'
        self.is_loaded = False
        self.face_cascade = None

        self.emotions = EMOTIONS

        # Healing colors - designed to improve mood, not match it
        self.healing_colors = {
            'Happy':    {'color': '#FFD700', 'emoji': '😊', 'message': 'Keeping your mood bright'},
            'Sad':      {'color': '#FF8C42', 'emoji': '😢', 'message': 'Warm light to lift your mood'},
            'Angry':    {'color': '#87CEEB', 'emoji': '😠', 'message': 'Cool blue to calm your mind'},
            'Fear':     {'color': '#FFF5E4', 'emoji': '😨', 'message': 'Bright light to remove darkness'},
            'Surprise': {'color': '#D8B4FE', 'emoji': '😲', 'message': 'Gentle color to calm your mind'},
            'Neutral':  {'color': '#FFFFFF', 'emoji': '😐', 'message': 'Natural comfortable light'},
            'Disgust':  {'color': '#98FFB3', 'emoji': '🤢', 'message': 'Fresh color to refresh your mood'}
        }

    def load_model(self, model_path=None):
        """Load the trained emotion model. Tries SVM first, then CNN."""
        # Load face cascade
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        models_dir = os.path.join(os.path.dirname(__file__), 'models')

        # Try SVM model first (works on all CPUs)
        svm_path = os.path.join(models_dir, 'emotion_svm.pkl')
        if os.path.exists(svm_path):
            try:
                import joblib
                self.model = joblib.load(svm_path)
                self.model_type = 'svm'
                self.is_loaded = True
                print("[Emotion Model] SVM model loaded successfully!")
                return True
            except Exception as e:
                print(f"[Emotion Model] SVM load error: {e}")

        # Try TensorFlow CNN model as fallback
        cnn_path = model_path or os.path.join(models_dir, 'emotion_model.h5')
        if os.path.exists(cnn_path):
            try:
                from tensorflow.keras.models import load_model
                self.model = load_model(cnn_path, compile=False)
                self.model_type = 'cnn'
                self.is_loaded = True
                print("[Emotion Model] CNN model loaded successfully!")
                return True
            except Exception as e:
                print(f"[Emotion Model] CNN load error: {e}")

        print("[Emotion Model] No model found. Run train_emotion_sklearn.py first!")
        self.is_loaded = False
        return False

    def train_model(self, dataset_path=None):
        """Train SVM model on FER2013 dataset."""
        if dataset_path is None:
            dataset_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'emotion face detection')

        try:
            # Use the sklearn training script
            from train_emotion_sklearn import train
            success = train()
            if success:
                # Reload model
                self.load_model()
            return success
        except Exception as e:
            print(f"[Emotion Model] Training error: {e}")
            return False

    def detect_emotion(self, image_data):
        """
        Detect emotion from image data.
        image_data: base64 encoded image string or numpy array
        """
        try:
            # Decode base64 image if string
            if isinstance(image_data, str):
                # Remove data URL prefix if present
                if ',' in image_data:
                    image_data = image_data.split(',')[1]
                img_bytes = base64.b64decode(image_data)
                nparr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            else:
                frame = image_data

            if frame is None:
                return {"error": "Could not decode image"}

            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h_img, w_img = gray.shape[:2]

            # Detect faces
            if self.face_cascade is None:
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                self.face_cascade = cv2.CascadeClassifier(cascade_path)

            # For small images (like FER2013 48x48), treat entire image as face
            is_small_image = max(h_img, w_img) <= 100

            if is_small_image:
                # Image is already a face crop — use it directly
                x, y, w, h = 0, 0, w_img, h_img
                face_roi = cv2.resize(gray, (48, 48))
            else:
                # Normal face detection for webcam/larger images
                faces = self.face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30)
                )

                if len(faces) == 0:
                    # Try with more relaxed params
                    faces = self.face_cascade.detectMultiScale(
                        gray, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20)
                    )

                if len(faces) == 0:
                    return {"error": "No face detected", "face_detected": False}

                # Use the largest face
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                face_roi = gray[y:y+h, x:x+w]
                face_roi = cv2.resize(face_roi, (48, 48))

            if self.is_loaded and self.model is not None:
                if self.model_type == 'svm':
                    # Use SVM model
                    features = extract_hog_features(face_roi)
                    features = features.reshape(1, -1)
                    prediction = self.model.predict(features)[0]
                    emotion = self.emotions[prediction]

                    # Get probability
                    proba = self.model.predict_proba(features)[0]
                    confidence = float(proba[prediction])

                elif self.model_type == 'cnn':
                    # Use CNN model
                    face_input = face_roi.astype('float32') / 255.0
                    face_input = face_input.reshape(1, 48, 48, 1)
                    predictions = self.model.predict(face_input, verbose=0)[0]
                    emotion_idx = int(np.argmax(predictions))
                    emotion = self.emotions[emotion_idx]
                    confidence = float(predictions[emotion_idx])
                else:
                    emotion = 'Neutral'
                    confidence = 0.5
            else:
                # Fallback: use pixel intensity analysis for demo
                mean_intensity = np.mean(face_roi)
                std_intensity = np.std(face_roi)

                if mean_intensity > 140:
                    emotion = 'Happy'
                elif mean_intensity < 80:
                    emotion = 'Sad'
                elif std_intensity > 60:
                    emotion = 'Surprise'
                elif std_intensity > 50:
                    emotion = 'Angry'
                else:
                    emotion = 'Neutral'
                confidence = 0.5

            healing = self.healing_colors[emotion]

            return {
                "face_detected": True,
                "emotion": emotion,
                "confidence": round(confidence, 2),
                "healing_color": healing['color'],
                "emoji": healing['emoji'],
                "message": healing['message'],
                "face_box": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}
            }

        except Exception as e:
            print(f"[Emotion Model] Detection error: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}


# Singleton instance
emotion_detector = EmotionDetector()
