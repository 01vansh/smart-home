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
import requests

# Fix Windows encoding
os.environ['PYTHONIOENCODING'] = 'utf-8'
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO

app = Flask(__name__)
app.config['SECRET_KEY'] = 'smart-home-ai-secret'
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
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


@app.route('/live_temperature', methods=['POST'])
def live_temperature():
    """Fetch live temperature from Open-Meteo API using device GPS coordinates."""
    try:
        data = request.get_json()
        lat = data.get('latitude')
        lon = data.get('longitude')

        if lat is None or lon is None:
            return jsonify({"success": False, "error": "Latitude and longitude are required"}), 400

        # Fetch current weather from Open-Meteo (free, no API key needed)
        weather_url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&current=temperature_2m,relative_humidity_2m,apparent_temperature,wind_speed_10m,weather_code"
            f"&timezone=auto"
        )
        weather_res = requests.get(weather_url, timeout=10)
        weather_data = weather_res.json()

        if 'current' not in weather_data:
            return jsonify({"success": False, "error": "Could not fetch weather data"}), 502

        current = weather_data['current']
        temperature = current.get('temperature_2m', 0)
        humidity = current.get('relative_humidity_2m', 0)
        apparent_temp = current.get('apparent_temperature', temperature)
        wind_speed = current.get('wind_speed_10m', 0)
        weather_code = current.get('weather_code', 0)
        timezone = weather_data.get('timezone', 'Unknown')

        # Reverse geocode for location name (Open-Meteo geocoding)
        location_name = f"{round(float(lat), 2)}°, {round(float(lon), 2)}°"
        try:
            geo_url = f"https://geocoding-api.open-meteo.com/v1/search?latitude={lat}&longitude={lon}&count=1"
            # Use a simple nominatim-style reverse geocode instead
            reverse_url = f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json&zoom=10"
            geo_res = requests.get(reverse_url, timeout=5, headers={'User-Agent': 'SmartHomeAI/1.0'})
            geo_data = geo_res.json()
            if 'address' in geo_data:
                addr = geo_data['address']
                city = addr.get('city', addr.get('town', addr.get('village', addr.get('state', ''))))
                country = addr.get('country', '')
                location_name = f"{city}, {country}" if city else location_name
        except Exception:
            pass  # Keep coordinate-based name if reverse geocoding fails

        # Map weather codes to descriptions
        weather_descriptions = {
            0: 'Clear Sky', 1: 'Mainly Clear', 2: 'Partly Cloudy', 3: 'Overcast',
            45: 'Foggy', 48: 'Rime Fog', 51: 'Light Drizzle', 53: 'Moderate Drizzle',
            55: 'Dense Drizzle', 61: 'Slight Rain', 63: 'Moderate Rain', 65: 'Heavy Rain',
            71: 'Slight Snow', 73: 'Moderate Snow', 75: 'Heavy Snow', 80: 'Rain Showers',
            81: 'Moderate Rain Showers', 82: 'Violent Rain Showers',
            95: 'Thunderstorm', 96: 'Thunderstorm with Hail', 99: 'Thunderstorm with Heavy Hail'
        }
        weather_desc = weather_descriptions.get(weather_code, 'Unknown')

        # Run fan prediction on the live temperature
        fan_result = fan_predictor.predict(temperature)

        return jsonify({
            "success": True,
            "temperature": round(temperature, 1),
            "humidity": humidity,
            "apparent_temperature": round(apparent_temp, 1),
            "wind_speed": round(wind_speed, 1),
            "weather_description": weather_desc,
            "weather_code": weather_code,
            "location": location_name,
            "timezone": timezone,
            "fan_prediction": fan_result
        })

    except requests.exceptions.Timeout:
        return jsonify({"success": False, "error": "Weather API timeout. Try again."}), 504
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


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
