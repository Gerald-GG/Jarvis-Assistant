import os
import time
import traceback
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# 1. Force Load Environment Variables
load_dotenv(override=True)

# 2. Modern LangChain 1.x Imports
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    print("⚠️ Packages missing. Run: pip install -U langchain-openai langchain-google-genai")
    LANGCHAIN_AVAILABLE = False

from utils.config import ConfigManager

class BaseLLM(ABC):
    @abstractmethod
    def generate_response(self, prompt: str, context: Optional[List] = None) -> str: pass
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]: pass
    @abstractmethod
    def is_available(self) -> bool: pass

class GoogleClient(BaseLLM):
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash-lite", temperature: float = 0.7):
        self.model_name = model
        try:
            self.llm = ChatGoogleGenerativeAI(
                model=model,
                google_api_key=api_key,
                temperature=temperature,
                max_output_tokens=2048,
                convert_system_message_to_human=True,
                timeout=30
            )
            self._available = True
        except Exception as e:
            print(f"❌ Gemini Initialization Failed: {e}")
            self._available = False

    def is_available(self) -> bool:
        return self._available
    
    def get_model_info(self) -> Dict:
        return {"provider": "google", "model": self.model_name, "available": self._available}

    def generate_response(self, prompt: str, context: Optional[List] = None) -> str:
        try:
            formatted_messages = []
            
            # 1. System Instruction - Modern 2025 approach
            system_instruction = "You are Jarvis, a helpful AI assistant. Respond concisely."
            
            # 2. History & Content Building
            if context:
                for msg in context[-5:]: # Reduced to last 5 for maximum stability
                    role = msg.get('role', 'user')
                    content = str(msg.get('content', ''))
                    if role == 'user':
                        formatted_messages.append(HumanMessage(content=content))
                    else:
                        formatted_messages.append(AIMessage(content=content))

            # 3. Final Human Message (Merging system instruction if list is empty)
            if not formatted_messages:
                # If no history, put the system prompt and user prompt together
                full_prompt = f"{system_instruction}\n\nUser: {prompt}"
                formatted_messages.append(HumanMessage(content=full_prompt))
            else:
                formatted_messages.append(HumanMessage(content=str(prompt)))
            
            # 4. Invoke the model
            response = self.llm.invoke(formatted_messages)
            return response.content
            
        except Exception as e:
            print(f"--- CRITICAL BRAIN ERROR ---")
            print(traceback.format_exc())
            return f"Jarvis Error: {str(e)}"

class OpenAIClient(BaseLLM):
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        self.model_name = model
        try:
            self.llm = ChatOpenAI(openai_api_key=api_key, model=model)
            self._available = True
        except:
            self._available = False

    def is_available(self) -> bool: return self._available
    def get_model_info(self) -> Dict: return {"provider": "openai", "model": self.model_name, "available": self._available}
    def generate_response(self, prompt: str, context: Optional[List] = None) -> str:
        try:
            return self.llm.invoke(str(prompt)).content
        except Exception as e:
            return f"OpenAI Error: {str(e)}"

class FallbackLLM(BaseLLM):
    def is_available(self) -> bool: return True
    def get_model_info(self) -> Dict: 
        return {"provider": "fallback", "model": "offline", "available": True}
    def generate_response(self, prompt: str, context: Optional[List] = None) -> str:
        return "I am in offline mode. Please ensure GOOGLE_API_KEY is set in your .env file."

class Brain:
    def __init__(self, config_manager: ConfigManager):
        self.config = getattr(config_manager, 'config', config_manager)
        self.conversation_history = []
        self.llm_client = self._initialize_llm()
        
        if self.llm_client:
            info = self.llm_client.get_model_info()
            provider = info.get('provider', 'unknown')
            model_name = info.get('model', 'unknown')
            print(f"✅ Brain initialized with {provider} ({model_name})")
        else:
            print("❌ Brain failed to initialize even in fallback mode.")

    def _initialize_llm(self) -> BaseLLM:
        llm_config = getattr(self.config, 'llm', None)
        provider = getattr(llm_config, 'provider', 'google').lower()
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")

        if not api_key or "sk-..." in api_key or "your_key" in api_key:
            print("⚠️ No valid API key found. Entering fallback mode.")
            return FallbackLLM()

        try:
            if provider == "google":
                return GoogleClient(api_key=api_key, model=getattr(llm_config, 'model', 'gemini-2.5-flash-lite'))
            elif provider == "openai":
                return OpenAIClient(api_key=api_key)
        except Exception as e:
            print(f"❌ Initialization Error: {e}")
            return FallbackLLM()
        
        return FallbackLLM()

    def process_query(self, query: str, context: Optional[dict] = None) -> str:
        """Fixed: Now accepts the context argument from api.py"""
        if not self.is_available():
            return "Jarvis is offline. Check your API configuration."
        
        # Use our internal conversation_history for the LLM context
        response = self.llm_client.generate_response(query, self.conversation_history)
        
        self.conversation_history.append({"role": "user", "content": query})
        self.conversation_history.append({"role": "assistant", "content": response})
        
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
            
        return response

    def clear_memory(self):
        """Added: Required by the /memory/clear endpoint in api.py"""
        self.conversation_history = []
        return True

    def is_available(self) -> bool:
        return self.llm_client is not None and self.llm_client.is_available()