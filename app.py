"""
Smart Home AI — Flask Main Server
Routes for all 4 features:
  1. Fan Speed Control
  2. Emotion Mood Lighting
  3. Face Recognition Door
  4. Voice Control
"""

import os
import sys
import json

# Fix Windows encoding
os.environ['PYTHONIOENCODING'] = 'utf-8'
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO

app = Flask(__name__)
app.config['SECRET_KEY'] = 'smart-home-ai-secret'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# ─── Import ML modules ───────────────────────────────────────────────
from fan_model import fan_predictor
from emotion import emotion_detector
from face_auth import face_authenticator
from voice import voice_controller

# ─── Initialize Models on Startup ────────────────────────────────────
def init_models():
    """Load/train all models when server starts."""
    print("\n" + "=" * 60)
    print("  Smart Home AI — Initializing Models")
    print("=" * 60 + "\n")

    # Feature 1: Fan Speed Predictor
    fan_predictor.train()

    # Feature 2: Emotion Detector
    emotion_detector.load_model()

    # Feature 3: Face Authentication
    face_authenticator.train()

    # Feature 4: Voice Controller
    voice_controller.load_vosk_model()
    # SpeechBrain loads on demand to save startup time

    print("\n" + "=" * 60)
    print("  All models initialized! Server ready.")
    print("=" * 60 + "\n")


# ═══════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════

@app.route('/')
def index():
    """Serve the single-page application."""
    return render_template('index.html')


# ─── Feature 1: Fan Speed ────────────────────────────────────────────

@app.route('/predict_fan', methods=['POST'])
def predict_fan():
    """Predict fan speed based on temperature."""
    try:
        data = request.get_json()
        temperature = data.get('temperature')

        if temperature is None:
            return jsonify({"success": False, "message": "Temperature is required", "error": "Temperature is required"}), 400

        result = fan_predictor.predict(temperature)
        if result:
            return jsonify(result)
        else:
            return jsonify({"success": False, "message": "Model not ready", "error": "Model not ready"}), 503
    except Exception as e:
        return jsonify({"success": False, "message": str(e), "error": str(e)}), 500


# ─── Feature 2: Emotion Detection ────────────────────────────────────

@app.route('/predict_emotion', methods=['POST'])
def predict_emotion():
    """Detect emotion from image."""
    try:
        data = request.get_json()
        image_data = data.get('image')

        if not image_data:
            return jsonify({"success": False, "message": "Image data is required", "error": "Image data is required"}), 400

        result = emotion_detector.detect_emotion(image_data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "message": str(e), "error": str(e)}), 500

@app.route('/train_emotion', methods=['POST'])
def train_emotion():
    """Train emotion detection model on FER2013 dataset."""
    try:
        success = emotion_detector.train_model()
        if success:
            return jsonify({"success": True, "message": "Emotion model trained successfully!"})
        else:
            return jsonify({"success": False, "message": "Training failed. Check console.", "error": "Training failed"}), 500
    except Exception as e:
        return jsonify({"success": False, "message": str(e), "error": str(e)}), 500


# ─── Feature 3: Face Recognition ─────────────────────────────────────

@app.route('/recognize_face', methods=['POST'])
def recognize_face():
    """Recognize face from webcam image."""
    try:
        data = request.get_json()
        image_data = data.get('image')

        if not image_data:
            return jsonify({"success": False, "message": "Image data is required", "error": "Image data is required"}), 400

        result = face_authenticator.recognize(image_data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "message": str(e), "error": str(e)}), 500

@app.route('/register_face', methods=['POST'])
def register_face():
    """Register a face photo for a person."""
    try:
        data = request.get_json()
        name = data.get('name')
        image_data = data.get('image')

        if not name or not image_data:
            return jsonify({"success": False, "message": "Name and image are required", "error": "Name and image are required"}), 400

        result = face_authenticator.add_face(name, image_data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "message": str(e), "error": str(e)}), 500

@app.route('/registered_faces', methods=['GET'])
def registered_faces():
    """Get list of registered faces."""
    try:
        people = face_authenticator.get_registered_people()
        return jsonify({"people": people})
    except Exception as e:
        return jsonify({"success": False, "message": str(e), "error": str(e)}), 500

@app.route('/retrain_faces', methods=['POST'])
def retrain_faces():
    """Retrain face recognition model."""
    try:
        success = face_authenticator.train()
        return jsonify({"success": success})
    except Exception as e:
        return jsonify({"success": False, "message": str(e), "error": str(e)}), 500

@app.route('/remove_face', methods=['POST'])
def remove_face():
    """Remove a registered person."""
    try:
        data = request.get_json()
        name = data.get('name')
        if not name:
            return jsonify({"success": False, "message": "Name is required", "error": "Name is required"}), 400
        result = face_authenticator.remove_person(name)
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "message": str(e), "error": str(e)}), 500

@app.route('/remove_all_faces', methods=['POST'])
def remove_all_faces():
    """Remove all registered faces."""
    try:
        result = face_authenticator.remove_all()
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "message": str(e), "error": str(e)}), 500


# ─── Feature 4: Voice Control ────────────────────────────────────────

@app.route('/device_status', methods=['GET'])
def device_status():
    """Get current device states."""
    try:
        return jsonify(voice_controller.get_device_states())
    except Exception as e:
        return jsonify({"success": False, "message": str(e), "error": str(e)}), 500

@app.route('/execute_command', methods=['POST'])
def execute_command():
    """Execute a device command from text."""
    try:
        data = request.get_json()
        text = data.get('text', '')

        if not text:
            return jsonify({"success": False, "message": "Command text is required", "error": "Command text is required"}), 400

        # Parse command
        command = voice_controller.parse_command(text)

        if not command.get("found", False):
            return jsonify({
                "success": False,
                "message": f"Unknown command: '{text}'",
                "error": "Unknown command",
                "text": text,
                "device_states": voice_controller.get_device_states()
            })

        # Execute
        result = voice_controller.execute_command(command["device"], command["action"])

        return jsonify({
            "success": True,
            "text": text,
            "device": command["device"],
            "action": command["action"],
            "message": result["message"],
            "device_states": voice_controller.get_device_states()
        })
    except Exception as e:
        return jsonify({"success": False, "message": str(e), "error": str(e)}), 500

@app.route('/verify_voice', methods=['POST'])
def verify_voice():
    """Verify if voice matches a registered person."""
    try:
        data = request.get_json()
        audio_data = data.get('audio')

        if not audio_data:
            return jsonify({"success": False, "message": "Audio data is required", "error": "Audio data is required"}), 400

        result = voice_controller.verify_speaker(audio_data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "message": str(e), "error": str(e)}), 500

@app.route('/process_voice', methods=['POST'])
def process_voice():
    """Full voice pipeline: verify → recognize → execute."""
    try:
        data = request.get_json()
        audio_data = data.get('audio')

        if not audio_data:
            return jsonify({"success": False, "message": "Audio data is required", "error": "Audio data is required"}), 400

        result = voice_controller.process_voice_command(audio_data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "message": str(e), "error": str(e)}), 500

@app.route('/registered_voices', methods=['GET'])
def registered_voices():
    """Get registered voice list."""
    try:
        voices = voice_controller.get_registered_voices()
        return jsonify({
            "voices": voices,
            "count": len(voices),
            "max": voice_controller.max_registrations
        })
    except Exception as e:
        return jsonify({"success": False, "message": str(e), "error": str(e)}), 500

@app.route('/register_voice', methods=['POST'])
def register_voice():
    """Register a new voice."""
    try:
        data = request.get_json()
        name = data.get('name')
        audio_data = data.get('audio')

        if not name or not audio_data:
            return jsonify({"success": False, "message": "Name and audio are required", "error": "Name and audio are required"}), 400

        result = voice_controller.register_voice(name, audio_data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "message": str(e), "error": str(e)}), 500

@app.route('/remove_voice', methods=['POST'])
def remove_voice():
    """Remove a registered voice."""
    try:
        data = request.get_json()
        name = data.get('name')
        if not name:
            return jsonify({"success": False, "message": "Name is required", "error": "Name is required"}), 400
        result = voice_controller.remove_voice(name)
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "message": str(e), "error": str(e)}), 500

@app.route('/remove_all_voices', methods=['POST'])
def remove_all_voices():
    """Remove all registered voices."""
    try:
        result = voice_controller.remove_all_voices()
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "message": str(e), "error": str(e)}), 500


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

# Initialize models at startup
init_models()

if __name__ == '__main__':
    print("\n🏠 Smart Home AI running at http://localhost:5001\n")
    socketio.run(app, host='0.0.0.0', port=5001, debug=False, allow_unsafe_werkzeug=True)
