"""
Feature 3: Face Recognition Security Door
Uses OpenCV LBPH (Local Binary Pattern Histogram)
Matches against known_faces folder

Key improvements:
- Saves preprocessed, consistent face crops (not raw frames)
- Blur detection on registration
- Histogram equalization for lighting consistency
- Augmented training (flipped + brightened versions)
- Clear quality feedback to user
"""

import cv2
import numpy as np
import os
import base64
import shutil


class FaceAuthenticator:
    FACE_SIZE = 200  # Standard face crop size

    def __init__(self):
        self.recognizer = None
        self.face_cascade = None
        self.is_trained = False
        self.label_names = {}  # {label_id: person_name}
        self.model_path = os.path.join(os.path.dirname(__file__), 'models', 'face_model.yml')
        self.known_faces_dir = os.path.join(os.path.dirname(__file__), 'known_faces')

        # Initialize face detector
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def _preprocess_face(self, face_gray):
        """Standard preprocessing for face ROI: resize + histogram equalize."""
        face = cv2.resize(face_gray, (self.FACE_SIZE, self.FACE_SIZE))
        face = cv2.equalizeHist(face)
        return face

    def _detect_face(self, gray_image):
        """Detect the largest face in a grayscale image."""
        faces = self.face_cascade.detectMultiScale(
            gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
        if len(faces) == 0:
            # Retry with relaxed params
            faces = self.face_cascade.detectMultiScale(
                gray_image, scaleFactor=1.05, minNeighbors=3, minSize=(40, 40)
            )
        if len(faces) == 0:
            return None
        return max(faces, key=lambda f: f[2] * f[3])

    def _check_blur(self, gray_face):
        """Check if face image is blurry. Returns (is_blurry, score)."""
        laplacian_var = cv2.Laplacian(gray_face, cv2.CV_64F).var()
        is_blurry = laplacian_var < 30  # Threshold for blur
        return is_blurry, round(laplacian_var, 1)

    def _augment_face(self, face_preprocessed):
        """Create augmented versions of a face for better training."""
        augmented = [face_preprocessed]

        # Horizontally flipped
        augmented.append(cv2.flip(face_preprocessed, 1))

        # Slightly brighter
        brighter = cv2.convertScaleAbs(face_preprocessed, alpha=1.15, beta=10)
        augmented.append(brighter)

        # Slightly darker
        darker = cv2.convertScaleAbs(face_preprocessed, alpha=0.85, beta=-10)
        augmented.append(darker)

        return augmented

    def train(self):
        """Train LBPH model from known_faces directory."""
        try:
            self.recognizer = cv2.face.LBPHFaceRecognizer_create(
                radius=2, neighbors=8, grid_x=8, grid_y=8, threshold=80.0
            )

            if not os.path.exists(self.known_faces_dir):
                os.makedirs(self.known_faces_dir)
                self.is_trained = False
                return False

            faces = []
            labels = []
            label_id = 0
            self.label_names = {}

            for person_name in sorted(os.listdir(self.known_faces_dir)):
                person_dir = os.path.join(self.known_faces_dir, person_name)
                if not os.path.isdir(person_dir):
                    continue

                person_faces = []
                for img_file in os.listdir(person_dir):
                    img_path = os.path.join(person_dir, img_file)
                    if not img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        continue

                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue

                    # Check if image is already a preprocessed face crop (200x200)
                    h, w = img.shape[:2]
                    if h == self.FACE_SIZE and w == self.FACE_SIZE:
                        face_preprocessed = cv2.equalizeHist(img)
                    else:
                        # Detect face in the image
                        face_box = self._detect_face(img)
                        if face_box is None:
                            continue
                        x, y, fw, fh = face_box
                        face_roi = img[y:y+fh, x:x+fw]
                        face_preprocessed = self._preprocess_face(face_roi)

                    # Augment each face
                    for aug_face in self._augment_face(face_preprocessed):
                        person_faces.append(aug_face.astype(np.uint8))

                if len(person_faces) > 0:
                    self.label_names[label_id] = person_name
                    for f in person_faces:
                        faces.append(f)
                        labels.append(label_id)
                    print(f"[Face Auth] Loaded {len(person_faces)} face(s) for '{person_name}' (with augmentation)")
                    label_id += 1

            if len(faces) == 0:
                print("[Face Auth] No faces found. Register faces via admin panel!")
                self.is_trained = False
                return False

            labels = np.array(labels)
            self.recognizer.train(faces, labels)

            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            self.recognizer.write(self.model_path)
            self.is_trained = True
            print(f"[Face Auth] Trained with {len(faces)} faces, {len(self.label_names)} people")
            return True

        except Exception as e:
            print(f"[Face Auth] Training error: {e}")
            import traceback
            traceback.print_exc()
            return False

    def recognize(self, image_data):
        """Recognize face from image data (base64 or numpy array)."""
        try:
            if not self.is_trained or self.recognizer is None:
                return {
                    "matched": False,
                    "message": "No registered faces. Register via admin panel first.",
                    "face_detected": False
                }

            # Decode image
            if isinstance(image_data, str):
                if ',' in image_data:
                    image_data = image_data.split(',')[1]
                img_bytes = base64.b64decode(image_data)
                nparr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            else:
                frame = image_data

            if frame is None:
                return {"error": "Could not decode image", "face_detected": False}

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_box = self._detect_face(gray)

            if face_box is None:
                return {
                    "matched": False,
                    "face_detected": False,
                    "message": "No face detected. Please look at the camera."
                }

            x, y, w, h = face_box
            face_roi = gray[y:y+h, x:x+w]
            face_preprocessed = self._preprocess_face(face_roi)

            label, confidence = self.recognizer.predict(face_preprocessed)

            # LBPH confidence = distance. Lower = closer match.
            # Webcam images have lighting/angle variation, so threshold must be generous
            # Known face:   typically 30-65
            # Unknown face:  typically 80-120+
            MATCH_THRESHOLD = 75

            person_name = self.label_names.get(label, "Unknown")
            import sys
            print(f"[Face Auth] Predict: label={label} ({person_name}), distance={confidence:.1f}, threshold={MATCH_THRESHOLD}", flush=True)
            sys.stdout.flush()

            if confidence < MATCH_THRESHOLD:
                match_quality = "High" if confidence < 40 else "Medium" if confidence < 60 else "Low"
                return {
                    "matched": True,
                    "face_detected": True,
                    "person_name": person_name,
                    "confidence": round(100 - confidence, 1),
                    "distance": round(confidence, 1),
                    "match_quality": match_quality,
                    "face_box": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
                    "message": f"Welcome {person_name}! Access Granted"
                }
            else:
                return {
                    "matched": False,
                    "face_detected": True,
                    "confidence": round(confidence, 1),
                    "distance": round(confidence, 1),
                    "face_box": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
                    "message": "Access Denied - Unknown Person"
                }

        except Exception as e:
            print(f"[Face Auth] Recognition error: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e), "face_detected": False}

    def get_registered_people(self):
        """Get list of registered people."""
        people = []
        if os.path.exists(self.known_faces_dir):
            for name in sorted(os.listdir(self.known_faces_dir)):
                path = os.path.join(self.known_faces_dir, name)
                if os.path.isdir(path):
                    count = len([f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
                    if count > 0:
                        people.append({"name": name, "photo_count": count})
        return people

    def add_face(self, person_name, image_data):
        """Add a validated face photo for a person. Saves preprocessed face crop."""
        try:
            person_name = person_name.lower().strip()
            if not person_name or len(person_name) < 2:
                return {"success": False, "message": "Name must be at least 2 characters"}

            person_dir = os.path.join(self.known_faces_dir, person_name)
            os.makedirs(person_dir, exist_ok=True)

            # Decode image
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            img_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                return {"success": False, "message": "Could not decode image"}

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect face
            face_box = self._detect_face(gray)
            if face_box is None:
                return {"success": False, "message": "No face detected! Look at the camera clearly and ensure good lighting."}

            x, y, w, h = face_box

            # Check face size
            if w < 60 or h < 60:
                return {"success": False, "message": "Face too small. Move closer to the camera."}

            # Extract face ROI
            face_roi = gray[y:y+h, x:x+w]

            # Check blur
            is_blurry, blur_score = self._check_blur(face_roi)
            if is_blurry:
                return {
                    "success": False,
                    "message": f"Image is too blurry (score: {blur_score}). Hold camera steady and ensure focus."
                }

            # Check brightness
            mean_val = np.mean(face_roi)
            if mean_val < 40:
                return {"success": False, "message": "Face is too dark. Please improve lighting."}
            if mean_val > 230:
                return {"success": False, "message": "Face is too bright/overexposed. Reduce lighting."}

            # Preprocess and save face crop (consistent 200x200 with histogram eq)
            face_preprocessed = self._preprocess_face(face_roi)

            existing = len([f for f in os.listdir(person_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            filename = f"photo{existing + 1}.jpg"
            filepath = os.path.join(person_dir, filename)

            # Save the preprocessed face crop
            cv2.imwrite(filepath, face_preprocessed)

            # Retrain model
            self.train()

            total = existing + 1
            if total >= 5:
                quality_msg = "Excellent accuracy!"
            elif total >= 3:
                quality_msg = "Good accuracy."
            else:
                quality_msg = f"Add {3 - total} more photo(s) for better accuracy."

            return {
                "success": True,
                "message": f"Face registered for '{person_name}' ({total} photos). {quality_msg}",
                "filename": filename,
                "photo_count": total,
                "blur_score": blur_score,
                "face_box": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}
            }

        except Exception as e:
            return {"success": False, "message": str(e)}

    def remove_person(self, person_name):
        """Remove a registered person and all their photos."""
        try:
            person_name = person_name.lower().strip()
            person_dir = os.path.join(self.known_faces_dir, person_name)

            if not os.path.exists(person_dir):
                return {"success": False, "message": f"Person '{person_name}' not found"}

            shutil.rmtree(person_dir)
            self.train()
            return {"success": True, "message": f"Removed '{person_name}' successfully"}

        except Exception as e:
            return {"success": False, "message": str(e)}

    def remove_all(self):
        """Remove all registered faces."""
        try:
            if os.path.exists(self.known_faces_dir):
                shutil.rmtree(self.known_faces_dir)
            os.makedirs(self.known_faces_dir, exist_ok=True)

            self.recognizer = None
            self.is_trained = False
            self.label_names = {}

            if os.path.exists(self.model_path):
                os.remove(self.model_path)

            return {"success": True, "message": "All faces removed"}

        except Exception as e:
            return {"success": False, "message": str(e)}


# Singleton instance
face_authenticator = FaceAuthenticator()
