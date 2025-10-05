"""
Voice alert utilities.

Uses pyttsx3 for offline TTS; if unavailable or for Tamil, uses gTTS as a fallback.
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


class VoiceAlert:
    def __init__(self, lang="en"):
        self.lang = lang
        if pyttsx3 is not None:
            try:
                self.engine = pyttsx3.init()
            except Exception:
                self.engine = None
        else:
            self.engine = None

    def _play_audio(self, path):
        try:
            playsound.playsound(path)
        except Exception:
            pass

    def speak(self, text, lang=None):
        """Speak text in the requested language.

        This method is non-blocking and spawns a thread for playback.
        """
        lang = lang or self.lang
        if self.engine is not None and lang.startswith("en"):
            # Use pyttsx3 for English for offline safety
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
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                    tts = gTTS(text=text, lang=lang[:2])
                    tts.write_to_fp(tmp)
                    tmp_path = tmp.name
                threading.Thread(target=self._play_audio, args=(tmp_path,), daemon=True).start()
                return
            except Exception:
                pass

        # Last fallback: print to console
        print(f"VOICE[{lang}]: {text}")
