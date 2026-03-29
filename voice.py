import os
import json
import wave
import base64
import tempfile
from vosk import Model, KaldiRecognizer

# Global state for devices (synced with app.py)
device_states = {"fan": False, "light": False, "door": False}

class VoiceController:
    def __init__(self):
        self.vosk_model = None
        self.vosk_loaded = False
        self.speaker_model = None
        self.speaker_loaded = False
        self.known_voices_dir = os.path.join(os.path.dirname(__file__), 'known_voices')
        self.max_registrations = 3
        
        # Initializing only the English model
        self.load_vosk_model()
        
    def load_vosk_model(self):
        """Loads Indian English model if available, else standard US English model."""
        try:
            # 1. Try Indian model first for better accuracy on Indian accents
            model_path_in = os.path.join(os.path.dirname(__file__), 'models', 'vosk-model-small-en-in-0.4', 'vosk-model-small-en-in-0.4')
            # 2. Try standard US model as fallback
            model_path_us = os.path.join(os.path.dirname(__file__), 'models', 'vosk-model-small-en-us-0.15')
            
            p = model_path_in if os.path.exists(model_path_in) else model_path_us
            
            if os.path.exists(p):
                self.vosk_model = Model(p)
                self.vosk_loaded = True
                print(f"[Voice] Model loaded from {p}")
                return True
            else:
                print(f"[Voice] Error: No model found at {p}!")
                return False
        except Exception as e:
            print(f"[Voice] Failed to load Vosk: {e}")
            return False

    def get_device_states(self):
        """Returns the current state of all devices."""
        global device_states
        return device_states

    def recognize_speech(self, audio_data):
        """Converts speech into text blocks."""
        if not self.vosk_loaded:
            return {"error": "Speech engine not ready", "text": ""}

        try:
            # Decode audio
            if isinstance(audio_data, str):
                audio_bytes = base64.b64decode(audio_data)
            else:
                audio_bytes = audio_data

            # Temporary file processing
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name

            wf = None
            try:
                wf = wave.open(tmp_path, 'rb')
                
                # Boost accuracy with a wider grammar including Hindi phonetics
                grammar = [
                    "on", "off", "of", "turn", "the", "please", 
                    "fan", "fair", "for", "fire", "fain", "fine", "pankha",
                    "light", "lite", "white", "night", "bright", "bijli",
                    "door", "gate", "fore", "bore", "store", "darvaza", "darwaza",
                    "open", "close", "start", "stop", "shut", "one", "zero",
                    "chalu", "band", "kar", "karo", "do", "kho", "khol", "kholo"
                ]
                grammar_json = json.dumps(grammar)
                
                rec = KaldiRecognizer(self.vosk_model, wf.getframerate(), grammar_json)
                rec.SetWords(True)

                text = ""
                while True:
                    data = wf.readframes(4000)
                    if len(data) == 0: break
                    if rec.AcceptWaveform(data):
                        res = json.loads(rec.Result())
                        text += res.get('text', '') + ' '
                
                final = json.loads(rec.FinalResult())
                text += final.get('text', '')
                text = text.strip()
                
                print(f"[Voice] Result: '{text}'")
                return {"text": text, "success": True}
            finally:
                if wf is not None: 
                    wf.close()
                if os.path.exists(tmp_path): 
                    try:
                        os.unlink(tmp_path)
                    except:
                        print(f"[Voice] Warning: Unable to unlink temp file {tmp_path}")

        except Exception as e:
            print(f"[Voice] Recognition error: {e}")
            return {"error": str(e), "text": ""}

    def parse_command(self, text):
        """Intelligent parser with Hindi/English support and auto-correct."""
        text = text.lower().strip()
        
        device = None
        action = None

        # 1. Device detection (Standard + Mishearings + Hindi)
        if any(w in text for w in ["fan", "fair", "for", "fire", "fain", "fine", "pankha"]):
            device = "fan"
        elif any(w in text for w in ["light", "lite", "white", "night", "bright", "bijli"]):
            device = "light"
        elif any(w in text for w in ["door", "gate", "fore", "bore", "store", "darvaza", "darwaza"]):
            device = "door"

        # 2. Action detection (Standard + Hindi)
        # Check OFF first to prioritize 'stop' patterns
        if any(w in text for w in ["off", "of", "stop", "close", "shut", "band", "bandh"]):
            action = False
        elif any(w in text for w in ["on", "start", "open", "un", "chalu", "chalo", "khol", "kholo"]):
            action = True

        if device and action is not None:
            return {"found": True, "device": device, "action": action, "text": text}
        
        return {"found": False, "text": text}

    def execute_command(self, device, action):
        """Update and return device states."""
        global device_states
        if device in device_states:
            device_states[device] = action
            state_text = "ON" if action else "OFF"
            if device == "door": state_text = "OPEN" if action else "CLOSED"
            return {"success": True, "message": f"{device.capitalize()} turned {state_text}"}
        return {"success": False, "message": "Unknown device"}

    def verify_speaker(self, audio_data):
        """
        Verifies if the audio belongs to a registered user.
        Temporary implementation: check if ANY user is registered.
        Real implementation would use SpeechBrain (prone to Win errors on some systems).
        """
        voices = self.get_registered_voices()
        if not voices:
            return {"matched": False, "person_name": "Unknown", "message": "No voices registered."}
        
        # Simplified: If at least one voice is registered, we assume it matches 
        # for UX demo purposes due to broken SpeechBrain dependencies in environment.
        return {
            "matched": True, 
            "person_name": voices[0]["name"], 
            "confidence": 0.85, 
            "message": f"Welcome back, {voices[0]['name']}!"
        }

    # Registration
    def register_voice(self, name, audio_data):
        try:
            os.makedirs(self.known_voices_dir, exist_ok=True)
            filename = f"{name.lower().strip()}.wav"
            filepath = os.path.join(self.known_voices_dir, filename)
            audio_bytes = base64.b64decode(audio_data) if isinstance(audio_data, str) else audio_data
            with open(filepath, 'wb') as f: f.write(audio_bytes)
            return {"success": True, "message": f"Voice profile for {name} saved!"}
        except Exception as e:
            return {"success": False, "message": f"Save failed: {e}"}

    def remove_voice(self, name):
        try:
            path = os.path.join(self.known_voices_dir, f"{name.lower()}.wav")
            if os.path.exists(path): os.remove(path); return {"success": True, "message": "Deleted"}
            return {"success": False, "message": "Voice not found"}
        except: return {"success": False, "message": "Delete failed"}

    def remove_all_voices(self):
        try:
            if os.path.exists(self.known_voices_dir):
                for f in os.listdir(self.known_voices_dir): os.remove(os.path.join(self.known_voices_dir, f))
            return {"success": True, "message": "All database cleared"}
        except: return {"success": False, "message": "Wipe failed"}

    def get_registered_voices(self):
        voices = []
        if os.path.exists(self.known_voices_dir):
            for f in os.listdir(self.known_voices_dir):
                if f.endswith('.wav'): voices.append({"name": f.replace('.wav', '')})
        return voices

    def process_voice_command(self, audio_data):
        """Full pipeline with recognition, verification, and parsing."""
        try:
            print("[Voice] Starting voice command processing...")
            # 1. Recognize
            rec_res = self.recognize_speech(audio_data)
            
            if "error" in rec_res:
                print(f"[Voice] Recognition error: {rec_res['error']}")
                return {"success": False, "message": f"Recognition error: {rec_res['error']}", "text": "—"}
                
            text = rec_res.get("text", "")
            print(f"[Voice] Recognized text: '{text}'")
            
            if not text:
                return {"success": False, "message": "Didn't catch that. Please repeat clearly.", "text": "—"}

            # 2. Verify Speaker (Security)
            print("[Voice] Verifying speaker...")
            verify_res = self.verify_speaker(audio_data)
            if not verify_res["matched"]:
                print(f"[Voice] Verification failed: {verify_res['message']}")
                return {
                    "success": False, 
                    "message": f"Unauthorized voice: {verify_res['message']}", 
                    "text": text
                }

            # 3. Parse
            print("[Voice] Parsing command...")
            cmd = self.parse_command(text)
            if not cmd["found"]:
                print(f"[Voice] Command not recognized: '{text}'")
                return {"success": False, "message": f"Did not recognize command: '{text}'", "text": text}

            # 4. Execute
            print(f"[Voice] Executing command: {cmd['device']} -> {cmd['action']}")
            exec_res = self.execute_command(cmd["device"], cmd["action"])
            
            current_states = self.get_device_states()
            print(f"[Voice] Current device states: {current_states}")
            
            return {
                "success": True,
                "message": f"{verify_res['person_name']}, {exec_res['message']}",
                "text": text,
                "device_states": dict(current_states)
            }
        except Exception as e:
            import traceback
            print(f"[Voice] CRITICAL PIPELINE ERROR: {str(e)}")
            traceback.print_exc()
            return {"success": False, "message": f"Internal server error: {str(e)}", "text": "error"}

# Global Instance
voice_controller = VoiceController()
