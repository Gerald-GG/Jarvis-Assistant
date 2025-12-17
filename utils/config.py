import yaml
import os
from typing import Dict, Any
from pydantic import BaseModel

class LLMConfig(BaseModel):
    provider: str = "openai"
    model: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    api_key: str = ""

class SpeechConfig(BaseModel):
    stt_engine: str = "whisper"
    tts_engine: str = "pyttsx3"
    language: str = "en-US"
    wake_word: str = "jarvis"

class AssistantConfig(BaseModel):
    name: str = "Jarvis"
    llm: LLMConfig = LLMConfig()
    speech: SpeechConfig = SpeechConfig()

class ConfigManager:
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self) -> AssistantConfig:
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                yaml_config = yaml.safe_load(f)
                return AssistantConfig(**yaml_config)
        return AssistantConfig()
    
    def save_config(self):
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config.dict(), f)
    
    def update_config(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        self.save_config()