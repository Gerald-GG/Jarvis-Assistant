import os
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.llms import Ollama
from utils.config import ConfigManager

class BaseLLM(ABC):
    @abstractmethod
    def generate_response(self, prompt: str, context: Optional[List] = None) -> str:
        pass

class OpenAIClient(BaseLLM):
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo", temperature: float = 0.7):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        
    def generate_response(self, prompt: str, context: Optional[List] = None) -> str:
        messages = []
        if context:
            messages.extend(context)
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"

class AnthropicClient(BaseLLM):
    def __init__(self, api_key: str, model: str = "claude-3-haiku-20240307", temperature: float = 0.7):
        from anthropic import Anthropic
        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.temperature = temperature
        
    def generate_response(self, prompt: str, context: Optional[List] = None) -> str:
        system_prompt = "You are Jarvis, a helpful AI assistant."
        if context:
            system_prompt += f"\nContext: {context}"
            
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                temperature=self.temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            return f"Error: {str(e)}"

class LocalLLM(BaseLLM):
    def __init__(self, model: str = "llama2"):
        from langchain_community.llms import Ollama
        self.llm = Ollama(model=model)
        self.memory = ConversationBufferMemory()
        
    def generate_response(self, prompt: str, context: Optional[List] = None) -> str:
        try:
            chain = ConversationChain(llm=self.llm, memory=self.memory)
            response = chain.run(input=prompt)
            return response
        except Exception as e:
            return f"Error: {str(e)}"

class Brain:
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager.config
        self.llm_client = self._initialize_llm()
        self.conversation_history = []
        
    def _initialize_llm(self) -> BaseLLM:
        llm_config = self.config.llm
        api_key = os.getenv(f"{llm_config.provider.upper()}_API_KEY", "")
        
        if llm_config.provider.lower() == "openai":
            return OpenAIClient(
                api_key=api_key,
                model=llm_config.model,
                temperature=llm_config.temperature
            )
        elif llm_config.provider.lower() == "anthropic":
            return AnthropicClient(
                api_key=api_key,
                model=llm_config.model,
                temperature=llm_config.temperature
            )
        elif llm_config.provider.lower() == "local":
            return LocalLLM(model=llm_config.model)
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_config.provider}")
    
    def process_query(self, query: str, user_context: Optional[Dict] = None) -> str:
        """Process a text query and return AI response"""
        # Add context to prompt if provided
        full_prompt = query
        if user_context:
            context_str = " ".join([f"{k}: {v}" for k, v in user_context.items()])
            full_prompt = f"Context: {context_str}\n\nUser: {query}"
        
        response = self.llm_client.generate_response(full_prompt, self.conversation_history)
        
        # Update conversation history
        self.conversation_history.append({"role": "user", "content": query})
        self.conversation_history.append({"role": "assistant", "content": response})
        
        # Keep only recent history
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
            
        return response
    
    def clear_memory(self):
        """Clear conversation history"""
        self.conversation_history = []