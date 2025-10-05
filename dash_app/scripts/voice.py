"""Voice alerts: try pyttsx3 for offline TTS; fallback to gTTS for Tamil/English when needed.
"""
import os
import tempfile
import threading
try:
    import pyttsx3
except Exception:
    pyttsx3 = None
try:
    from gtts import gTTS
except Exception:
    gTTS = None
import playsound


class VoiceEngine:
    def __init__(self):
        self.engine = None
        if pyttsx3 is not None:
            try:
                self.engine = pyttsx3.init()
            except Exception:
                self.engine = None

    def _play_file(self, path):
        try:
            playsound.playsound(path)
        except Exception:
            pass

    def speak(self, text, lang='en'):
        # Non-blocking: prefer pyttsx3 for offline English; otherwise gTTS
        if self.engine is not None and lang.startswith('en'):
            def _s():
                try:
                    self.engine.say(text)
                    self.engine.runAndWait()
                except Exception:
                    pass
            threading.Thread(target=_s, daemon=True).start()
            return

        if gTTS is not None:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as f:
                    tmp = f.name
                tts = gTTS(text=text, lang=lang[:2])
                tts.save(tmp)
                threading.Thread(target=self._play_file, args=(tmp,), daemon=True).start()
                return
            except Exception:
                pass

        # Last fallback: print to console
        print(f"VOICE[{lang}]: {text}")


_VOICE = VoiceEngine()


def speak(text, lang='en'):
    _VOICE.speak(text, lang=lang)
