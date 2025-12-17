import os
import tempfile
import speech_recognition as sr
import pyttsx3
import whisper
from abc import ABC, abstractmethod
from typing import Optional, BinaryIO
from utils.config import ConfigManager

class BaseSTT(ABC):
    @abstractmethod
    def transcribe(self, audio_data: BinaryIO) -> str:
        pass

class WhisperSTT(BaseSTT):
    def __init__(self, model_size: str = "base"):
        self.model = whisper.load_model(model_size)
        
    def transcribe(self, audio_data: BinaryIO) -> str:
        # Save audio to temp file for Whisper
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_data.read())
            tmp_path = tmp.name
            
        try:
            result = self.model.transcribe(tmp_path, fp16=False)
            return result["text"]
        finally:
            os.unlink(tmp_path)

class GoogleSTT(BaseSTT):
    def __init__(self, language: str = "en-US"):
        self.recognizer = sr.Recognizer()
        self.language = language
        
    def transcribe(self, audio_data: BinaryIO) -> str:
        # Convert audio_data to AudioData for SpeechRecognition
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_data.read())
            tmp_path = tmp.name
            
        try:
            with sr.AudioFile(tmp_path) as source:
                audio = self.recognizer.record(source)
                text = self.recognizer.recognize_google(audio, language=self.language)
                return text
        except sr.UnknownValueError:
            return ""
        except sr.RequestError as e:
            return f"STT Error: {str(e)}"
        finally:
            os.unlink(tmp_path)

class BaseTTS(ABC):
    @abstractmethod
    def speak(self, text: str):
        pass

class PyTTSx3TTS(BaseTTS):
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        self.engine.setProperty('volume', 0.9)
        
    def speak(self, text: str):
        self.engine.say(text)
        self.engine.runAndWait()

class ElevenLabsTTS(BaseTTS):
    def __init__(self, api_key: str, voice_id: str = "Rachel"):
        from elevenlabs import generate, play, set_api_key
        set_api_key(api_key)
        self.generate = generate
        self.play = play
        self.voice_id = voice_id
        
    def speak(self, text: str):
        audio = self.generate(text=text, voice=self.voice_id)
        self.play(audio)

class SpeechProcessor:
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager.config
        self.stt_engine = self._initialize_stt()
        self.tts_engine = self._initialize_tts()
        self.recognizer = sr.Recognizer()
        
    def _initialize_stt(self) -> BaseSTT:
        stt_config = self.config.speech
        if stt_config.stt_engine.lower() == "whisper":
            return WhisperSTT()
        elif stt_config.stt_engine.lower() == "google":
            return GoogleSTT(language=stt_config.language)
        else:
            raise ValueError(f"Unsupported STT engine: {stt_config.stt_engine}")
    
    def _initialize_tts(self) -> BaseTTS:
        tts_config = self.config.speech
        if tts_config.tts_engine.lower() == "pyttsx3":
            return PyTTSx3TTS()
        elif tts_config.tts_engine.lower() == "elevenlabs":
            api_key = os.getenv("ELEVENLABS_API_KEY", "")
            return ElevenLabsTTS(api_key=api_key)
        else:
            raise ValueError(f"Unsupported TTS engine: {tts_config.tts_engine}")
    
    def listen_from_mic(self, timeout: int = 5, phrase_time_limit: int = 10) -> Optional[str]:
        """Listen from microphone and return transcribed text"""
        with sr.Microphone() as source:
            print("Listening...")
            try:
                audio = self.recognizer.listen(
                    source, 
                    timeout=timeout,
                    phrase_time_limit=phrase_time_limit
                )
                
                # Convert AudioData to bytes for STT engine
                audio_data = audio.get_wav_data()
                return self.stt_engine.transcribe(BytesIO(audio_data))
                
            except sr.WaitTimeoutError:
                print("No speech detected")
                return None
            except Exception as e:
                print(f"Error listening: {e}")
                return None
    
    def speak_text(self, text: str):
        """Convert text to speech"""
        print(f"Speaking: {text}")
        self.tts_engine.speak(text)
    
    def process_audio_file(self, audio_path: str) -> str:
        """Transcribe audio from file"""
        with open(audio_path, 'rb') as f:
            return self.stt_engine.transcribe(f)