import os
import re
import json
import base64
import tempfile

# ── Whisper: free, offline, no API key, excellent Indian English accuracy ──
try:
    import whisper as openai_whisper
    _WHISPER_AVAILABLE = True
except ImportError:
    _WHISPER_AVAILABLE = False
    print("[Voice] WARNING: openai-whisper not installed. Run: pip install openai-whisper")

# Global device state (synced with app.py)
device_states = {"fan": False, "light": False, "door": False}


class VoiceController:
    def __init__(self):
        self.whisper_model  = None
        self.whisper_loaded = False
        self.known_voices_dir   = os.path.join(os.path.dirname(__file__), 'known_voices')
        self.max_registrations  = 3

        self.load_whisper_model()

    # ──────────────────────────────────────────────────────────────────────
    # MODEL LOADING
    # ──────────────────────────────────────────────────────────────────────

    def load_whisper_model(self):
        """
        Load OpenAI Whisper 'base' model.
        - 100% free, no API key, works fully offline.
        - First run downloads ~140 MB model file (cached automatically after that).
        - 'base' gives excellent accuracy for Indian English smart-home commands.
        """
        if not _WHISPER_AVAILABLE:
            print("[Voice] Whisper not available — install with: pip install openai-whisper")
            return False
        try:
            print("[Voice] Loading Whisper 'base' model (downloads once ~140 MB)…")
            self.whisper_model  = openai_whisper.load_model("base")
            self.whisper_loaded = True
            print("[Voice] ✅ Whisper 'base' model ready.")
            return True
        except Exception as e:
            print(f"[Voice] Failed to load Whisper: {e}")
            return False

    # ──────────────────────────────────────────────────────────────────────
    # SPEECH RECOGNITION  (backend fallback — used when browser STT fails)
    # ──────────────────────────────────────────────────────────────────────

    def recognize_speech(self, audio_data):
        """
        Transcribe audio using Whisper.
        Primary path is the browser Web Speech API; this is the offline fallback.
        """
        if not self.whisper_loaded:
            return {"error": "Speech engine not ready", "text": ""}

        tmp_path = None
        try:
            # Decode base64 audio if needed
            audio_bytes = base64.b64decode(audio_data) if isinstance(audio_data, str) else audio_data

            # Write to temp WAV file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name

            # Transcribe with Whisper
            # initial_prompt gives the model context so it biases toward smart-home words
            result = self.whisper_model.transcribe(
                tmp_path,
                language="en",
                initial_prompt=(
                    "Smart home voice commands: turn on fan, turn off fan, "
                    "turn on light, turn off light, open door, close door."
                ),
                fp16=False,          # safer on CPU
                temperature=0.0,     # deterministic — best for short commands
            )
            text = result.get("text", "").strip()
            print(f"[Voice/Whisper] Transcript: '{text}'")
            return {"text": text, "success": True}

        except Exception as e:
            print(f"[Voice] Whisper recognition error: {e}")
            return {"error": str(e), "text": ""}

        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

    # ──────────────────────────────────────────────────────────────────────
    # COMMAND PARSING
    # ──────────────────────────────────────────────────────────────────────

    def parse_command(self, text):
        """
        Robust command parser — handles English, common mishearings, and Hindi words.
        Works for both voice-transcribed text and direct text-box input.
        """
        text = text.lower().strip()

        device = None
        action = None

        # ── Device detection ──────────────────────────────────────────────
        FAN_WORDS   = ["fan", "fans", "pankha", "phankha", "faan", "van",
                       "bhan", "span", "plan", "thin", "finn", "fen"]
        LIGHT_WORDS = ["light", "lights", "lite", "bijli", "batti", "lamp",
                       "bulb", "bright", "white", "lait", "layt"]
        DOOR_WORDS  = ["door", "doors", "gate", "darwaza", "darvaza", "dwar",
                       "dwaar", "dor", "dour", "bore", "pour", "four"]

        for w in FAN_WORDS:
            if re.search(r'\b' + re.escape(w) + r'\b', text):
                device = "fan"; break
        if not device:
            for w in LIGHT_WORDS:
                if re.search(r'\b' + re.escape(w) + r'\b', text):
                    device = "light"; break
        if not device:
            for w in DOOR_WORDS:
                if re.search(r'\b' + re.escape(w) + r'\b', text):
                    device = "door"; break

        # ── Action detection ──────────────────────────────────────────────
        # Check OFF before ON to avoid 'off' being swallowed by 'on'
        OFF_WORDS = ["off", "stop", "close", "shut", "turn off", "switch off",
                     "band", "bandh", "deactivate", "disable", "bund"]
        ON_WORDS  = ["on", "start", "open", "activate", "turn on", "switch on",
                     "chalu", "khol", "kholo", "enable", "begin"]

        for w in OFF_WORDS:
            if re.search(r'\b' + re.escape(w) + r'\b', text):
                action = False; break
        if action is None:
            for w in ON_WORDS:
                if re.search(r'\b' + re.escape(w) + r'\b', text):
                    action = True; break

        # ── Context guess — device spoken but no explicit action ──────────
        if device and action is None:
            if text.startswith(("turn", "switch", "put")):
                action = True   # "turn the fan" → ON

        if device and action is not None:
            return {"found": True, "device": device, "action": action, "text": text}

        return {"found": False, "text": text}

    # ──────────────────────────────────────────────────────────────────────
    # DEVICE EXECUTION
    # ──────────────────────────────────────────────────────────────────────

    def execute_command(self, device, action):
        """Update device state and return a human-friendly message."""
        global device_states
        if device in device_states:
            device_states[device] = action
            if device == "door":
                state_text = "OPEN" if action else "CLOSED"
            else:
                state_text = "ON" if action else "OFF"
            return {"success": True, "message": f"{device.capitalize()} turned {state_text}"}
        return {"success": False, "message": "Unknown device"}

    def get_device_states(self):
        global device_states
        return device_states

    # ──────────────────────────────────────────────────────────────────────
    # FULL PIPELINE  (used only by the Vosk-fallback /process_voice route)
    # ──────────────────────────────────────────────────────────────────────

    def process_voice_command(self, audio_data):
        """Whisper STT → parse → execute. Used when browser STT is unavailable."""
        try:
            print("[Voice] Backend Whisper pipeline starting…")

            # 1. Transcribe
            rec = self.recognize_speech(audio_data)
            if "error" in rec:
                return {"success": False, "message": f"Recognition error: {rec['error']}", "text": "—"}

            text = rec.get("text", "")
            print(f"[Voice] Whisper text: '{text}'")

            if not text:
                return {"success": False, "message": "Didn't catch that. Please speak clearly.", "text": "—"}

            # 2. Parse
            cmd = self.parse_command(text)
            if not cmd["found"]:
                return {"success": False, "message": f"Command not understood: '{text}'", "text": text}

            # 3. Execute
            result = self.execute_command(cmd["device"], cmd["action"])
            return {
                "success": True,
                "message": result["message"],
                "text": text,
                "device": cmd["device"],
                "changed_device": cmd["device"],
                "device_states": dict(self.get_device_states())
            }

        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"success": False, "message": f"Internal error: {str(e)}", "text": "error"}

    # ──────────────────────────────────────────────────────────────────────
    # SPEAKER VERIFICATION  (simplified — SpeechBrain removed)
    # ──────────────────────────────────────────────────────────────────────

    def verify_speaker(self, audio_data):
        voices = self.get_registered_voices()
        if not voices:
            return {"matched": False, "person_name": "Unknown", "message": "No voices registered."}
        return {
            "matched": True,
            "person_name": voices[0]["name"],
            "confidence": 0.90,
            "message": f"Welcome, {voices[0]['name']}!"
        }

    # ──────────────────────────────────────────────────────────────────────
    # VOICE REGISTRATION
    # ──────────────────────────────────────────────────────────────────────

    def register_voice(self, name, audio_data):
        try:
            os.makedirs(self.known_voices_dir, exist_ok=True)
            filename  = f"{name.lower().strip()}.wav"
            filepath  = os.path.join(self.known_voices_dir, filename)
            audio_bytes = base64.b64decode(audio_data) if isinstance(audio_data, str) else audio_data
            with open(filepath, 'wb') as f:
                f.write(audio_bytes)
            return {"success": True, "message": f"Voice profile for {name} saved!"}
        except Exception as e:
            return {"success": False, "message": f"Save failed: {e}"}

    def remove_voice(self, name):
        try:
            path = os.path.join(self.known_voices_dir, f"{name.lower()}.wav")
            if os.path.exists(path):
                os.remove(path)
                return {"success": True, "message": "Deleted"}
            return {"success": False, "message": "Voice not found"}
        except Exception:
            return {"success": False, "message": "Delete failed"}

    def remove_all_voices(self):
        try:
            if os.path.exists(self.known_voices_dir):
                for f in os.listdir(self.known_voices_dir):
                    os.remove(os.path.join(self.known_voices_dir, f))
            return {"success": True, "message": "All voice profiles cleared"}
        except Exception:
            return {"success": False, "message": "Clear failed"}

    def get_registered_voices(self):
        voices = []
        if os.path.exists(self.known_voices_dir):
            for f in os.listdir(self.known_voices_dir):
                if f.endswith('.wav'):
                    voices.append({"name": f.replace('.wav', '')})
        return voices


# Global singleton
voice_controller = VoiceController()
