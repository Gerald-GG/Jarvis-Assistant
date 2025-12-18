import os
import tempfile
import traceback
import speech_recognition as sr
import pyttsx3
import whisper
from io import BytesIO
from abc import ABC, abstractmethod
from typing import Optional, BinaryIO, Dict
from utils.config import ConfigManager

class BaseSTT(ABC):
    @abstractmethod
    def transcribe(self, audio_data: BinaryIO) -> str: pass

class WhisperSTT(BaseSTT):
    def __init__(self, model_size: str = "base"):
        print(f"📦 Loading Whisper model ({model_size})...")
        self.model = whisper.load_model(model_size)
        
    def transcribe(self, audio_data: BinaryIO) -> str:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_data.read())
            tmp_path = tmp.name
            
        try:
            result = self.model.transcribe(tmp_path, fp16=False)
            return result["text"].strip()
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

class GoogleSTT(BaseSTT):
    def __init__(self, language: str = "en-US"):
        self.recognizer = sr.Recognizer()
        self.language = language
        
    def transcribe(self, audio_data: BinaryIO) -> str:
        try:
            with sr.AudioFile(audio_data) as source:
                audio = self.recognizer.record(source)
                return self.recognizer.recognize_google(audio, language=self.language)
        except (sr.UnknownValueError, sr.RequestError):
            return ""

class BaseTTS(ABC):
    @abstractmethod
    def speak(self, text: str): pass

class PyTTSx3TTS(BaseTTS):
    def __init__(self):
        self.engine = pyttsx3.init()
        self._setup_voice()
        
    def _setup_voice(self):
        voices = self.engine.getProperty('voices')
        self.engine.setProperty('rate', 175) 
        self.engine.setProperty('volume', 1.0)
        
        for voice in voices:
            if "david" in voice.name.lower() or "alex" in voice.name.lower():
                self.engine.setProperty('voice', voice.id)
                break

    def speak(self, text: str):
        self.engine.say(text)
        self.engine.runAndWait()

class ElevenLabsTTS(BaseTTS):
    def __init__(self, api_key: str, voice_id: str = "George"):
        try:
            from elevenlabs.client import ElevenLabs
            from elevenlabs import play
            self.client = ElevenLabs(api_key=api_key)
            self.play = play
            self.voice_id = voice_id
        except ImportError:
            print("❌ ElevenLabs SDK missing. Run: pip install elevenlabs")

    def speak(self, text: str):
        try:
            audio = self.client.generate(
                text=text,
                voice=self.voice_id,
                model="eleven_multilingual_v2"
            )
            self.play(audio)
        except Exception as e:
            print(f"ElevenLabs Error: {e}")

class SpeechProcessor:
    def __init__(self, config_manager: ConfigManager):
        self.config = getattr(config_manager, 'config', config_manager)
        self.recognizer = sr.Recognizer()
        self.stt_engine = self._initialize_stt()
        self.tts_engine = self._initialize_tts()
        
    def _initialize_stt(self) -> BaseSTT:
        stt_config = self.config.speech
        engine_type = stt_config.stt_engine.lower()
        
        if engine_type == "whisper":
            return WhisperSTT()
        elif engine_type == "google":
            return GoogleSTT(language=stt_config.language)
        else:
            raise ValueError(f"Unsupported STT engine: {engine_type}")
    
    def _initialize_tts(self) -> BaseTTS:
        tts_config = self.config.speech
        engine_type = tts_config.tts_engine.lower()
        
        if engine_type == "pyttsx3":
            return PyTTSx3TTS()
        elif engine_type == "elevenlabs":
            api_key = os.getenv("ELEVENLABS_API_KEY", "")
            return ElevenLabsTTS(api_key=api_key)
        else:
            raise ValueError(f"Unsupported TTS engine: {engine_type}")
    
    def listen_from_mic(self) -> Optional[str]:
        """Local microphone capture optimized for Linux/PulseAudio."""
        # device_index=4 is 'pulse' based on your list_mics.py output
        try:
            with sr.Microphone(device_index=4) as source:
                print("Listening (via PulseAudio)...")
                
                # Calibrate for background noise (increased to 2s for Linux stability)
                self.recognizer.adjust_for_ambient_noise(source, duration=2.0)
                
                # Manual sensitivity settings
                self.recognizer.energy_threshold = 150 
                self.recognizer.dynamic_energy_threshold = False
                
                audio = self.recognizer.listen(source, timeout=7, phrase_time_limit=10)
                audio_data = BytesIO(audio.get_wav_data())
                return self.stt_engine.transcribe(audio_data)
        except Exception as e:
            print(f"🎙️ Mic Error: {e}")
            return None
    
    def speak_text(self, text: str):
        if not text: return
        print(f"🎙️ Jarvis: {text}")
        self.tts_engine.speak(text)