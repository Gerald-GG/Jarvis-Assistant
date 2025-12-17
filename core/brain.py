import os
import time
import json
from typing import List, Dict, Any, Optional, Union
from abc import ABC, abstractmethod
import traceback

# Try to import LangChain components with fallbacks
try:
    from langchain.memory import ConversationBufferMemory
    from langchain.chains import ConversationChain
    LANGCHAIN_AVAILABLE = True
except ImportError:
    print("⚠️  LangChain not available, using fallback conversation memory")
    LANGCHAIN_AVAILABLE = False

try:
    from langchain_openai import ChatOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    print("⚠️  langchain-openai not available")
    OPENAI_AVAILABLE = False

try:
    from langchain_anthropic import ChatAnthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    print("⚠️  langchain-anthropic not available")
    ANTHROPIC_AVAILABLE = False

try:
    from langchain_community.llms import Ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    print("⚠️  Ollama integration not available")
    OLLAMA_AVAILABLE = False

from utils.config import ConfigManager

class BaseLLM(ABC):
    @abstractmethod
    def generate_response(self, prompt: str, context: Optional[List] = None) -> str:
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the LLM is available/configured"""
        pass

class OpenAIClient(BaseLLM):
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo", temperature: float = 0.7, max_tokens: int = 1000):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = None
        self._initialize_client()
        
    def _initialize_client(self):
        """Initialize OpenAI client with error handling"""
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
            self._client_available = True
        except ImportError:
            print("⚠️  OpenAI Python package not installed")
            self._client_available = False
        except Exception as e:
            print(f"⚠️  Failed to initialize OpenAI client: {e}")
            self._client_available = False
    
    def is_available(self) -> bool:
        """Check if OpenAI client is available"""
        return self._client_available and self.api_key and len(self.api_key.strip()) > 0
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the OpenAI model"""
        model_info = {
            "provider": "openai",
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "available": self.is_available()
        }
        
        # Add model-specific capabilities
        if "gpt-4" in self.model:
            model_info["capabilities"] = ["chat", "reasoning", "function_calling"]
            model_info["context_window"] = 128000 if "turbo" in self.model else 8192
        elif "gpt-3.5" in self.model:
            model_info["capabilities"] = ["chat", "function_calling"]
            model_info["context_window"] = 16385
        
        return model_info
        
    def generate_response(self, prompt: str, context: Optional[List] = None) -> str:
        """Generate response using OpenAI API"""
        if not self.is_available():
            return "OpenAI client not available. Please check your API key and configuration."
        
        messages = []
        
        # Add system message
        system_message = "You are Jarvis, a helpful and intelligent AI assistant. Provide clear, concise, and accurate responses."
        messages.append({"role": "system", "content": system_message})
        
        # Add context if provided
        if context:
            for item in context:
                if isinstance(item, dict) and "role" in item and "content" in item:
                    messages.append(item)
        
        # Add user prompt
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            error_msg = str(e)
            if "authentication" in error_msg.lower() or "api key" in error_msg.lower():
                return "Authentication error: Please check your OpenAI API key in config.yaml"
            elif "quota" in error_msg.lower():
                return "API quota exceeded: Please check your OpenAI account usage"
            elif "rate limit" in error_msg.lower():
                return "Rate limit exceeded: Please wait a moment before trying again"
            else:
                return f"OpenAI API error: {error_msg}"

class AnthropicClient(BaseLLM):
    def __init__(self, api_key: str, model: str = "claude-3-haiku-20240307", temperature: float = 0.7, max_tokens: int = 1000):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = None
        self._initialize_client()
        
    def _initialize_client(self):
        """Initialize Anthropic client with error handling"""
        try:
            from anthropic import Anthropic
            self.client = Anthropic(api_key=self.api_key)
            self._client_available = True
        except ImportError:
            print("⚠️  Anthropic Python package not installed")
            self._client_available = False
        except Exception as e:
            print(f"⚠️  Failed to initialize Anthropic client: {e}")
            self._client_available = False
    
    def is_available(self) -> bool:
        """Check if Anthropic client is available"""
        return self._client_available and self.api_key and len(self.api_key.strip()) > 0
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the Anthropic model"""
        model_info = {
            "provider": "anthropic",
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "available": self.is_available()
        }
        
        # Add model-specific capabilities
        if "claude-3" in self.model:
            model_info["capabilities"] = ["chat", "vision", "reasoning"]
            model_info["context_window"] = 200000
            if "opus" in self.model:
                model_info["capabilities"].append("advanced_reasoning")
        
        return model_info
        
    def generate_response(self, prompt: str, context: Optional[List] = None) -> str:
        """Generate response using Anthropic API"""
        if not self.is_available():
            return "Anthropic client not available. Please check your API key and configuration."
        
        # Build system prompt
        system_prompt = "You are Jarvis, a helpful and intelligent AI assistant. Provide clear, concise, and accurate responses."
        
        # Add context to system prompt
        if context:
            context_text = "\n".join([f"{item.get('role', 'unknown')}: {item.get('content', '')}" 
                                     for item in context if isinstance(item, dict)])
            system_prompt = f"{system_prompt}\n\nContext from conversation:\n{context_text}"
            
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            error_msg = str(e)
            if "authentication" in error_msg.lower() or "api key" in error_msg.lower():
                return "Authentication error: Please check your Anthropic API key in config.yaml"
            elif "quota" in error_msg.lower():
                return "API quota exceeded: Please check your Anthropic account usage"
            elif "rate limit" in error_msg.lower():
                return "Rate limit exceeded: Please wait a moment before trying again"
            else:
                return f"Anthropic API error: {error_msg}"

class LocalLLM(BaseLLM):
    def __init__(self, model: str = "llama2", base_url: Optional[str] = None, temperature: float = 0.7):
        self.model = model
        self.temperature = temperature
        self.base_url = base_url
        self.llm = None
        self.memory = None
        self._initialize_llm()
        
    def _initialize_llm(self):
        """Initialize local LLM with error handling"""
        try:
            if OLLAMA_AVAILABLE:
                self.llm = Ollama(model=self.model, base_url=self.base_url, temperature=self.temperature)
                self._llm_available = True
                
                # Initialize memory if LangChain is available
                if LANGCHAIN_AVAILABLE:
                    self.memory = ConversationBufferMemory()
                else:
                    # Fallback memory
                    self.memory = {"history": []}
                    print("⚠️  Using fallback memory for local LLM")
            else:
                print("⚠️  Ollama not available for local LLM")
                self._llm_available = False
        except Exception as e:
            print(f"⚠️  Failed to initialize local LLM: {e}")
            self._llm_available = False
    
    def is_available(self) -> bool:
        """Check if local LLM is available"""
        return self._llm_available and self.llm is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the local model"""
        return {
            "provider": "local",
            "model": self.model,
            "temperature": self.temperature,
            "available": self.is_available(),
            "capabilities": ["chat", "completion"],
            "context_window": 4096  # Default for local models
        }
        
    def generate_response(self, prompt: str, context: Optional[List] = None) -> str:
        """Generate response using local LLM"""
        if not self.is_available():
            return "Local LLM not available. Please make sure Ollama is running and the model is installed."
        
        try:
            # Use LangChain if available
            if LANGCHAIN_AVAILABLE and self.memory:
                chain = ConversationChain(llm=self.llm, memory=self.memory)
                response = chain.run(input=prompt)
                return response
            else:
                # Fallback: direct call to Ollama
                # This is a simplified version - you might need to adjust based on your Ollama setup
                import requests
                
                # Prepare the request to Ollama API
                ollama_url = self.base_url or "http://localhost:11434"
                api_url = f"{ollama_url.rstrip('/')}/api/generate"
                
                payload = {
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.temperature
                    }
                }
                
                response = requests.post(api_url, json=payload)
                
                if response.status_code == 200:
                    result = response.json()
                    return result.get("response", "No response from Ollama")
                else:
                    return f"Ollama API error: {response.status_code} - {response.text}"
                    
        except Exception as e:
            error_msg = str(e)
            if "connection" in error_msg.lower():
                return "Cannot connect to local LLM. Make sure Ollama is running and the model is installed."
            else:
                return f"Local LLM error: {error_msg}"

class FallbackLLM(BaseLLM):
    """Fallback LLM that provides basic responses when no AI is available"""
    
    def __init__(self):
        self.responses = {
            "greeting": [
                "Hello! I'm Jarvis, your AI assistant.",
                "Hi there! I'm here to help.",
                "Greetings! I'm Jarvis, ready to assist you."
            ],
            "help": [
                "I can help you with questions, calculations, and general assistance.",
                "You can ask me anything! I'm here to help.",
                "I'm your AI assistant. Try asking me a question!"
            ],
            "error": "I'm currently running in limited mode. Please check your AI configuration in config.yaml.",
            "unknown": "I received your message, but I'm running in fallback mode. Please configure an AI provider in config.yaml."
        }
        
        self.simple_responses = {
            "hello": "Hello! How can I help you today?",
            "hi": "Hi there!",
            "how are you": "I'm functioning normally, thank you for asking!",
            "what can you do": "I can answer questions, help with tasks, and assist with information. Try asking me something!",
            "thank you": "You're welcome!",
            "bye": "Goodbye! Have a great day!"
        }
    
    def is_available(self) -> bool:
        """Fallback is always available"""
        return True
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the fallback model"""
        return {
            "provider": "fallback",
            "model": "basic_response",
            "available": True,
            "capabilities": ["basic_responses"],
            "context_window": 0
        }
    
    def generate_response(self, prompt: str, context: Optional[List] = None) -> str:
        """Generate a simple fallback response"""
        prompt_lower = prompt.lower().strip()
        
        # Check for simple queries first
        for key, response in self.simple_responses.items():
            if key in prompt_lower:
                return response
        
        # Check for greeting
        if any(word in prompt_lower for word in ["hello", "hi", "hey", "greetings"]):
            import random
            return random.choice(self.responses["greeting"])
        
        # Check for help request
        if any(word in prompt_lower for word in ["help", "what can you do", "capabilities"]):
            import random
            return random.choice(self.responses["help"])
        
        # Default response
        if "error" in prompt_lower or "problem" in prompt_lower:
            return self.responses["error"]
        
        return self.responses["unknown"]

class Brain:
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager.config
        self.llm_client = None
        self.conversation_history = []
        self.performance_stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "total_processing_time": 0,
            "avg_response_time": 0
        }
        
        # Initialize LLM client
        self._initialize_llm()
        
        # Print initialization status
        if self.llm_client and self.llm_client.is_available():
            model_info = self.llm_client.get_model_info()
            print(f"✅ Brain initialized with {model_info['provider']}:{model_info['model']}")
        else:
            print("⚠️  Brain initialized in fallback mode - using basic responses")
    
    def _initialize_llm(self) -> BaseLLM:
        """Initialize the appropriate LLM client based on configuration"""
        llm_config = self.config.llm
        
        # Try to get API key from config first, then environment
        api_key = None
        if hasattr(llm_config, 'api_key') and llm_config.api_key:
            api_key = llm_config.api_key
        else:
            # Fallback to environment variable
            api_key = os.getenv(f"{llm_config.provider.upper()}_API_KEY", "")
        
        provider = llm_config.provider.lower()
        
        try:
            if provider == "openai":
                if not OPENAI_AVAILABLE:
                    print("⚠️  OpenAI provider selected but langchain-openai not available")
                    return FallbackLLM()
                
                if not api_key:
                    print("⚠️  OpenAI API key not found in config.yaml or environment")
                    return FallbackLLM()
                
                return OpenAIClient(
                    api_key=api_key,
                    model=llm_config.model,
                    temperature=llm_config.temperature
                )
                
            elif provider == "anthropic":
                if not ANTHROPIC_AVAILABLE:
                    print("⚠️  Anthropic provider selected but langchain-anthropic not available")
                    return FallbackLLM()
                
                if not api_key:
                    print("⚠️  Anthropic API key not found in config.yaml or environment")
                    return FallbackLLM()
                
                return AnthropicClient(
                    api_key=api_key,
                    model=llm_config.model,
                    temperature=llm_config.temperature
                )
                
            elif provider == "local":
                if not OLLAMA_AVAILABLE:
                    print("⚠️  Local provider selected but Ollama integration not available")
                    return FallbackLLM()
                
                # Check for Ollama base URL in config
                base_url = None
                if hasattr(llm_config, 'base_url') and llm_config.base_url:
                    base_url = llm_config.base_url
                
                return LocalLLM(
                    model=llm_config.model,
                    base_url=base_url,
                    temperature=llm_config.temperature
                )
                
            else:
                print(f"⚠️  Unsupported LLM provider: {provider}")
                return FallbackLLM()
                
        except Exception as e:
            print(f"⚠️  Error initializing LLM client: {e}")
            return FallbackLLM()
    
    def process_query(self, query: str, user_context: Optional[Dict] = None) -> str:
        """Process a text query and return AI response"""
        start_time = time.time()
        self.performance_stats["total_queries"] += 1
        
        # Validate input
        if not query or not isinstance(query, str) or len(query.strip()) == 0:
            return "Please provide a valid query."
        
        query = query.strip()
        
        # Check for special commands
        if query.lower() in ["/help", "/commands", "/?"]:
            return self._get_help_response()
        
        if query.lower() in ["/status", "/health", "/ping"]:
            return self._get_status_response()
        
        if query.lower() in ["/clear", "/reset"]:
            self.clear_memory()
            return "Conversation memory cleared."
        
        if query.lower().startswith("/model "):
            return self._handle_model_command(query)
        
        # Prepare prompt with context
        full_prompt = self._prepare_prompt(query, user_context)
        
        try:
            # Generate response
            response = self.llm_client.generate_response(full_prompt, self.conversation_history)
            
            # Update conversation history
            self._update_conversation_history(query, response)
            
            # Update performance stats
            processing_time = time.time() - start_time
            self.performance_stats["total_processing_time"] += processing_time
            self.performance_stats["successful_queries"] += 1
            self.performance_stats["avg_response_time"] = (
                self.performance_stats["total_processing_time"] / 
                self.performance_stats["successful_queries"]
            )
            
            return response
            
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            print(f"❌ {error_msg}")
            
            # Return a helpful error message
            if "API key" in str(e) or "authentication" in str(e):
                return "Authentication error: Please check your API key in config.yaml"
            elif "quota" in str(e):
                return "API quota exceeded. Please check your account usage."
            elif "rate limit" in str(e):
                return "Rate limit exceeded. Please wait a moment before trying again."
            else:
                return f"I encountered an error: {str(e)}. Please try again or check the logs for details."
    
    def _prepare_prompt(self, query: str, user_context: Optional[Dict] = None) -> str:
        """Prepare the prompt with context"""
        if not user_context:
            return query
        
        # Format context as key-value pairs
        context_parts = []
        for key, value in user_context.items():
            if isinstance(value, (str, int, float, bool)):
                context_parts.append(f"{key}: {value}")
            elif isinstance(value, dict):
                context_parts.append(f"{key}: {json.dumps(value)}")
        
        if context_parts:
            context_str = "\n".join(context_parts)
            return f"Context:\n{context_str}\n\nUser query: {query}"
        
        return query
    
    def _update_conversation_history(self, query: str, response: str):
        """Update conversation history with new exchange"""
        # Add to history
        self.conversation_history.append({"role": "user", "content": query})
        self.conversation_history.append({"role": "assistant", "content": response})
        
        # Limit history size (keep last 20 exchanges = 40 messages)
        max_history = getattr(self.config.memory, 'max_history', 10) * 2
        if len(self.conversation_history) > max_history:
            self.conversation_history = self.conversation_history[-max_history:]
    
    def _get_help_response(self) -> str:
        """Generate help response"""
        model_info = self.llm_client.get_model_info() if self.llm_client else {}
        
        help_text = "I'm Jarvis, your AI assistant. Here's what I can do:\n\n"
        help_text += "• Answer questions and provide information\n"
        help_text += "• Help with calculations and problem-solving\n"
        help_text += "• Maintain conversation context\n"
        help_text += "• Process text queries through AI\n\n"
        
        if model_info.get('provider') != 'fallback':
            help_text += f"Current AI provider: {model_info.get('provider', 'unknown')}\n"
            help_text += f"Current model: {model_info.get('model', 'unknown')}\n\n"
        
        help_text += "Available commands:\n"
        help_text += "• /help or /? - Show this help message\n"
        help_text += "• /status or /ping - Check system status\n"
        help_text += "• /clear or /reset - Clear conversation memory\n"
        help_text += "• /model info - Show current model information\n"
        
        return help_text
    
    def _get_status_response(self) -> str:
        """Generate status response"""
        model_info = self.llm_client.get_model_info() if self.llm_client else {}
        
        status_text = "Jarvis System Status:\n\n"
        status_text += f"• AI Provider: {model_info.get('provider', 'fallback')}\n"
        status_text += f"• Model: {model_info.get('model', 'basic_response')}\n"
        status_text += f"• Available: {'✅ Yes' if model_info.get('available', False) else '❌ No'}\n"
        status_text += f"• Total Queries: {self.performance_stats['total_queries']}\n"
        status_text += f"• Successful Queries: {self.performance_stats['successful_queries']}\n"
        
        if self.performance_stats['successful_queries'] > 0:
            status_text += f"• Avg Response Time: {self.performance_stats['avg_response_time']:.2f}s\n"
        
        status_text += f"• Conversation History: {len(self.conversation_history)//2} exchanges\n"
        
        return status_text
    
    def _handle_model_command(self, query: str) -> str:
        """Handle model-related commands"""
        if not self.llm_client:
            return "No LLM client available."
        
        command = query.lower().replace("/model", "").strip()
        
        if command == "info":
            model_info = self.llm_client.get_model_info()
            info_text = "Current Model Information:\n\n"
            
            for key, value in model_info.items():
                if key == "available":
                    info_text += f"• {key}: {'✅ Yes' if value else '❌ No'}\n"
                else:
                    info_text += f"• {key}: {value}\n"
            
            return info_text
        
        return f"Unknown model command: {command}. Try '/model info'"
    
    def clear_memory(self):
        """Clear conversation history and reset performance stats"""
        self.conversation_history = []
        
        # Reset conversation-related stats only
        self.performance_stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "total_processing_time": 0,
            "avg_response_time": 0
        }
        
        # Also clear LangChain memory if available
        if self.llm_client and hasattr(self.llm_client, 'memory'):
            if isinstance(self.llm_client.memory, ConversationBufferMemory):
                self.llm_client.memory.clear()
            elif isinstance(self.llm_client.memory, dict) and "history" in self.llm_client.memory:
                self.llm_client.memory["history"] = []
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return self.performance_stats.copy()
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get conversation summary"""
        return {
            "total_exchanges": len(self.conversation_history) // 2,
            "recent_queries": [msg["content"] for msg in self.conversation_history if msg["role"] == "user"][-5:],
            "memory_enabled": len(self.conversation_history) > 0
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        if self.llm_client:
            return self.llm_client.get_model_info()
        return {"provider": "unknown", "available": False}
    
    def is_available(self) -> bool:
        """Check if the brain is available and functional"""
        if not self.llm_client:
            return False
        return self.llm_client.is_available()