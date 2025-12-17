
   from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Generator
import uvicorn
import io
import tempfile
import os
import json
import asyncio
from datetime import datetime
import base64
import time
import traceback
from pathlib import Path

# Local imports
from core.brain import Brain
from core.speech import SpeechProcessor
from utils.config import ConfigManager

# Initialize FastAPI app
app = FastAPI(
    title="Jarvis Assistant API",
    description="A smart AI assistant with speech capabilities",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development only - restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global initialization (with error handling)
config_manager = None
brain = None
speech_processor = None

def initialize_components():
    """Initialize Jarvis components with retry logic"""
    global config_manager, brain, speech_processor
    
    max_retries = 3
    retry_delay = 2  # seconds
    
    for attempt in range(max_retries):
        try:
            print(f"🔄 Initializing Jarvis components (Attempt {attempt + 1}/{max_retries})...")
            
            config_manager = ConfigManager()
            brain = Brain(config_manager)
            speech_processor = SpeechProcessor(config_manager)
            
            print("✅ Jarvis components initialized successfully")
            
            # Log configuration
            if config_manager and config_manager.config:
                config_summary = {
                    "assistant_name": config_manager.config.assistant.name,
                    "llm_provider": config_manager.config.llm.provider,
                    "llm_model": config_manager.config.llm.model,
                    "stt_engine": config_manager.config.speech.stt_engine,
                    "tts_engine": config_manager.config.speech.tts_engine
                }
                print(f"📋 Configuration: {json.dumps(config_summary, indent=2)}")
            
            return True
            
        except Exception as e:
            print(f"⚠️  Initialization attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print(f"⏳ Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print("❌ All initialization attempts failed")
                print("⚠️  Running in limited mode - some features may not work")
                print(f"🔍 Error details: {traceback.format_exc()}")
                return False

# Request/Response Models
class QueryRequest(BaseModel):
    text: str
    context: Optional[dict] = None
    stream: Optional[bool] = False
    model: Optional[str] = None

class QueryResponse(BaseModel):
    query: str
    response: str
    timestamp: str
    processing_time: Optional[float] = None
    model_used: Optional[str] = None

class AudioResponse(BaseModel):
    text: str
    response: str
    timestamp: str
    audio_path: Optional[str] = None
    audio_url: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str
    components: Dict[str, bool]
    uptime: Optional[float] = None

class AudioTranscriptionRequest(BaseModel):
    language: Optional[str] = "en"
    task: Optional[str] = "transcribe"
    model_size: Optional[str] = "base"

class ModelInfo(BaseModel):
    id: str
    name: str
    provider: str
    max_tokens: Optional[int] = None
    context_window: Optional[int] = None
    capabilities: Optional[List[str]] = None

class VoiceInfo(BaseModel):
    id: str
    name: str
    language: str
    gender: Optional[str] = None
    sample_rate: Optional[int] = None

# Global variables
startup_time = datetime.now()

# ==================== STARTUP/SHUTDOWN EVENTS ====================

@app.on_event("startup")
async def startup_event():
    """Run on application startup"""
    print(f"\n{'='*60}")
    print(f"🚀 Jarvis Assistant API starting up at {datetime.now()}")
    print(f"{'='*60}")
    
    # Initialize components
    initialize_components()
    
    print(f"\n📡 API Documentation available at:")
    print(f"   - Swagger UI: http://localhost:8000/docs")
    print(f"   - ReDoc: http://localhost:8000/redoc")
    print(f"\n🎯 Health check: http://localhost:8000/health")
    print(f"{'='*60}\n")

@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown"""
    print(f"\n{'='*60}")
    print(f"🛑 Jarvis Assistant API shutting down at {datetime.now()}")
    print(f"⏱️  Uptime: {(datetime.now() - startup_time).total_seconds():.1f} seconds")
    print(f"{'='*60}")

# ==================== HEALTH & STATUS ENDPOINTS ====================

@app.get("/", tags=["Status"], response_model=Dict[str, Any])
async def root():
    """Root endpoint - API information"""
    uptime = (datetime.now() - startup_time).total_seconds()
    
    return {
        "message": "Jarvis Assistant API",
        "status": "running",
        "version": "1.0.0",
        "uptime_seconds": uptime,
        "startup_time": startup_time.isoformat(),
        "endpoints": [
            {"path": "/docs", "method": "GET", "description": "API documentation (Swagger UI)"},
            {"path": "/redoc", "method": "GET", "description": "API documentation (ReDoc)"},
            {"path": "/health", "method": "GET", "description": "System health check"},
            {"path": "/status", "method": "GET", "description": "Detailed system status"},
            {"path": "/models", "method": "GET", "description": "Available AI models"},
            {"path": "/voices", "method": "GET", "description": "Available TTS voices"},
            {"path": "/query", "method": "POST", "description": "Process text queries"},
            {"path": "/transcribe", "method": "POST", "description": "Transcribe audio to text"},
            {"path": "/speak", "method": "POST", "description": "Convert text to speech"},
            {"path": "/audio-query", "method": "POST", "description": "Full audio processing pipeline"},
            {"path": "/config", "method": "GET", "description": "Get current configuration"},
            {"path": "/config/update", "method": "POST", "description": "Update configuration"},
            {"path": "/memory/clear", "method": "POST", "description": "Clear conversation memory"}
        ]
    }

@app.get("/health", response_model=HealthResponse, tags=["Status"])
async def health():
    """Health check endpoint"""
    components_status = {
        "config_manager": config_manager is not None,
        "brain": brain is not None and hasattr(brain, 'llm_client') and brain.llm_client is not None,
        "speech_processor": speech_processor is not None,
        "fastapi": True,
        "uvicorn": True
    }
    
    # Check if brain is properly initialized
    brain_status = False
    if brain is not None:
        try:
            # Quick test to see if brain responds
            if hasattr(brain, 'process_query'):
                test_response = brain.process_query("test", {"health_check": True})
                brain_status = test_response is not None and len(test_response) > 0
        except:
            brain_status = False
    
    components_status["brain"] = brain_status
    
    status = "healthy" if all(components_status.values()) else "degraded"
    uptime = (datetime.now() - startup_time).total_seconds()
    
    return HealthResponse(
        status=status,
        version="1.0.0",
        timestamp=datetime.now().isoformat(),
        components=components_status,
        uptime=uptime
    )

@app.get("/status", tags=["Status"], response_model=Dict[str, Any])
async def status():
    """Detailed system status"""
    try:
        config = config_manager.config.dict() if config_manager else {}
        uptime = (datetime.now() - startup_time).total_seconds()
        
        # Get memory info if available
        memory_info = {}
        if brain and hasattr(brain, 'conversation_history'):
            memory_info = {
                "history_length": len(brain.conversation_history),
                "memory_enabled": True
            }
        
        # Get component versions if possible
        component_versions = {}
        try:
            import fastapi
            import uvicorn as uv
            component_versions["fastapi"] = fastapi.__version__
            component_versions["uvicorn"] = uv.__version__
        except:
            pass
        
        return {
            "status": "operational",
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": uptime,
            "components": {
                "brain": {
                    "status": "active" if brain else "inactive",
                    "model": config.get("llm", {}).get("model", "unknown") if config else "unknown",
                    "provider": config.get("llm", {}).get("provider", "unknown") if config else "unknown"
                },
                "speech": {
                    "status": "active" if speech_processor else "inactive",
                    "stt_engine": config.get("speech", {}).get("stt_engine", "unknown") if config else "unknown",
                    "tts_engine": config.get("speech", {}).get("tts_engine", "unknown") if config else "unknown"
                },
                "config": {
                    "status": "loaded" if config_manager else "missing",
                    "assistant_name": config.get("assistant", {}).get("name", "unknown") if config else "unknown"
                }
            },
            "memory": memory_info,
            "versions": component_versions,
            "endpoints_available": [
                "/health", "/status", "/models", "/voices", "/query", 
                "/transcribe", "/speak", "/audio-query", "/config"
            ]
        }
    except Exception as e:
        return {"status": "error", "message": str(e), "timestamp": datetime.now().isoformat()}

# ==================== AI & QUERY ENDPOINTS ====================

@app.post("/query", response_model=QueryResponse, tags=["AI"])
async def process_query(request: QueryRequest):
    """Process text query and return AI response"""
    start_time = datetime.now()
    
    if not brain:
        raise HTTPException(
            status_code=503,
            detail="Brain component not available. Check initialization."
        )
    
    try:
        # Override model if specified in request
        original_model = None
        if request.model and config_manager:
            original_model = config_manager.config.llm.model
            config_manager.config.llm.model = request.model
        
        response = brain.process_query(request.text, request.context)
        
        # Restore original model if changed
        if original_model and config_manager:
            config_manager.config.llm.model = original_model
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Get current model
        current_model = "unknown"
        if config_manager and config_manager.config:
            current_model = config_manager.config.llm.model
        
        return QueryResponse(
            query=request.text,
            response=response,
            timestamp=datetime.now().isoformat(),
            processing_time=processing_time,
            model_used=current_model
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )

@app.post("/query/stream", tags=["AI"])
async def process_query_stream(request: QueryRequest):
    """Process text query with streaming response (SSE)"""
    if not brain:
        raise HTTPException(status_code=503, detail="Brain component not available")
    
    async def stream_response():
        """Generator for streaming response"""
        try:
            # For now, simulate streaming by sending chunks
            response = brain.process_query(request.text, request.context)
            
            # Split response into chunks for simulation
            words = response.split()
            for i, word in enumerate(words):
                chunk = {
                    "chunk": word + " ",
                    "index": i,
                    "is_final": i == len(words) - 1
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                await asyncio.sleep(0.05)  # Simulate processing delay
                
            yield f"event: complete\ndata: {{\"message\": \"Stream complete\"}}\n\n"
            
        except Exception as e:
            error_chunk = {
                "error": str(e),
                "is_final": True
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
    
    return StreamingResponse(
        stream_response(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

# ==================== SPEECH ENDPOINTS ====================

@app.post("/transcribe", tags=["Speech"], response_model=Dict[str, Any])
async def transcribe_audio(
    file: UploadFile = File(...),
    request: Optional[AudioTranscriptionRequest] = None
):
    """Transcribe audio file to text"""
    if not speech_processor:
        raise HTTPException(
            status_code=503,
            detail="Speech processor not available. Check initialization."
        )
    
    try:
        # Validate file type
        allowed_types = ["audio/wav", "audio/mpeg", "audio/mp3", "audio/ogg", "audio/flac", "audio/x-wav", "application/octet-stream"]
        if file.content_type not in allowed_types:
            # Check file extension as fallback
            filename = file.filename.lower()
            allowed_extensions = ['.wav', '.mp3', '.ogg', '.flac', '.m4a', '.aac']
            if not any(filename.endswith(ext) for ext in allowed_extensions):
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type. Allowed: {', '.join(allowed_types)}"
                )
        
        # Read audio data
        audio_data = await file.read()
        
        if len(audio_data) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file")
        
        # Create temporary file for processing
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_data)
            tmp_path = tmp.name
        
        try:
            start_time = time.time()
            
            # Transcribe using speech processor
            if hasattr(speech_processor, 'process_audio_file'):
                text = speech_processor.process_audio_file(tmp_path)
            elif hasattr(speech_processor, 'stt_engine'):
                with open(tmp_path, 'rb') as f:
                    text = speech_processor.stt_engine.transcribe(io.BytesIO(f.read()))
            else:
                raise HTTPException(status_code=500, detail="Speech processor doesn't have transcription capability")
            
            processing_time = time.time() - start_time
            
            return {
                "text": text,
                "filename": file.filename,
                "content_type": file.content_type,
                "size_bytes": len(audio_data),
                "processing_time": processing_time,
                "timestamp": datetime.now().isoformat(),
                "success": True
            }
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
                
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error transcribing audio: {str(e)}"
        )

@app.post("/speak", tags=["Speech"], response_model=Dict[str, Any])
async def text_to_speech(
    text: str,
    background_tasks: BackgroundTasks,
    voice: Optional[str] = "default",
    speed: Optional[float] = 1.0,
    return_audio: Optional[bool] = False
):
    """Convert text to speech and return audio file"""
    if not speech_processor:
        raise HTTPException(
            status_code=503,
            detail="Speech processor not available. Check initialization."
        )
    
    if not text or len(text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        start_time = time.time()
        
        if return_audio:
            # Create temporary audio file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, delete_on_close=False) as tmp:
                tmp_path = tmp.name
            
            # For now, we'll simulate file creation
            # In a real implementation, you would:
            # 1. Generate audio file at tmp_path using speech_processor
            # 2. Return it with FileResponse
            # 3. Schedule cleanup with background_tasks
            
            # Placeholder: Just speak the text
            speech_processor.speak_text(text)
            
            processing_time = time.time() - start_time
            
            # Schedule cleanup
            def cleanup_temp_file(path: str):
                if os.path.exists(path):
                    os.unlink(path)
            
            background_tasks.add_task(cleanup_temp_file, tmp_path)
            
            return {
                "status": "success",
                "message": "Speech generated (audio file placeholder)",
                "text": text,
                "processing_time": processing_time,
                "timestamp": datetime.now().isoformat(),
                "audio_path": tmp_path,
                "audio_url": f"/audio/{os.path.basename(tmp_path)}" if return_audio else None
            }
        else:
            # Just speak without returning audio file
            speech_processor.speak_text(text)
            
            processing_time = time.time() - start_time
            
            return {
                "status": "success",
                "message": "Speech generated",
                "text": text,
                "processing_time": processing_time,
                "timestamp": datetime.now().isoformat()
            }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating speech: {str(e)}"
        )

@app.post("/audio-query", response_model=AudioResponse, tags=["Speech", "AI"])
async def process_audio_query(
    file: UploadFile = File(...),
    return_audio: Optional[bool] = False
):
    """Process audio query end-to-end: STT -> LLM -> TTS"""
    start_time = time.time()
    
    if not brain or not speech_processor:
        raise HTTPException(
            status_code=503,
            detail="Required components not available"
        )
    
    try:
        # Step 1: Transcribe audio
        audio_data = await file.read()
        
        if len(audio_data) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file")
        
        # Create temporary file for transcription
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_data)
            tmp_path = tmp.name
        
        try:
            # Transcribe
            if hasattr(speech_processor, 'stt_engine'):
                with open(tmp_path, 'rb') as f:
                    query_text = speech_processor.stt_engine.transcribe(io.BytesIO(f.read()))
            else:
                raise HTTPException(status_code=500, detail="Speech processor doesn't have transcription capability")
            
            # Step 2: Process with LLM
            response_text = brain.process_query(query_text)
            
            # Step 3: Generate speech
            speech_processor.speak_text(response_text)
            
            processing_time = time.time() - start_time
            
            return AudioResponse(
                text=query_text,
                response=response_text,
                timestamp=datetime.now().isoformat(),
                audio_path=None,  # In future, return path to generated audio file
                audio_url=None
            )
            
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing audio query: {str(e)}"
        )

# ==================== CONFIGURATION ENDPOINTS ====================

@app.get("/config", tags=["Configuration"], response_model=Dict[str, Any])
async def get_config():
    """Get current configuration"""
    if not config_manager:
        raise HTTPException(status_code=503, detail="Config manager not available")
    
    try:
        config_dict = config_manager.config.dict()
        
        # Mask sensitive information
        if "llm" in config_dict and "api_key" in config_dict["llm"]:
            if config_dict["llm"]["api_key"]:
                config_dict["llm"]["api_key"] = "***" + config_dict["llm"]["api_key"][-4:] if len(config_dict["llm"]["api_key"]) > 4 else "***"
        
        return {
            "config": config_dict,
            "config_path": config_manager.config_path if hasattr(config_manager, 'config_path') else "unknown",
            "timestamp": datetime.now().isoformat(),
            "config_file_exists": os.path.exists(config_manager.config_path) if hasattr(config_manager, 'config_path') else False
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading config: {str(e)}")

@app.post("/config/update", tags=["Configuration"], response_model=Dict[str, Any])
async def update_config(updates: dict):
    """Update configuration"""
    if not config_manager:
        raise HTTPException(status_code=503, detail="Config manager not available")
    
    try:
        # Save original config for comparison
        original_config = config_manager.config.dict()
        
        # Apply updates
        config_manager.update_config(**updates)
        
        # Get updated config
        updated_config = config_manager.config.dict()
        
        # Reinitialize components if critical config changed
        critical_keys = ["llm.provider", "llm.model", "llm.api_key", "speech.stt_engine", "speech.tts_engine"]
        
        needs_reinit = False
        for key in critical_keys:
            parts = key.split('.')
            orig_val = original_config
            upd_val = updated_config
            for part in parts:
                orig_val = orig_val.get(part, {}) if isinstance(orig_val, dict) else orig_val
                upd_val = upd_val.get(part, {}) if isinstance(upd_val, dict) else upd_val
            
            if orig_val != upd_val:
                needs_reinit = True
                break
        
        if needs_reinit:
            print("🔄 Critical configuration changed, reinitializing components...")
            initialize_components()
        
        return {
            "status": "success",
            "message": "Configuration updated" + (" and components reinitialized" if needs_reinit else ""),
            "config": config_manager.config.dict(),
            "reinitialized": needs_reinit,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating config: {str(e)}")

@app.post("/memory/clear", tags=["AI"], response_model=Dict[str, Any])
async def clear_memory():
    """Clear conversation memory"""
    if not brain:
        raise HTTPException(status_code=503, detail="Brain component not available")
    
    try:
        history_length = 0
        if hasattr(brain, 'conversation_history'):
            history_length = len(brain.conversation_history)
        
        brain.clear_memory()
        
        return {
            "status": "success",
            "message": "Conversation memory cleared",
            "cleared_items": history_length,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing memory: {str(e)}")

# ==================== UTILITY ENDPOINTS ====================

@app.get("/voices", tags=["Speech"], response_model=Dict[str, Any])
async def list_available_voices():
    """List available TTS voices"""
    if not speech_processor:
        raise HTTPException(status_code=503, detail="Speech processor not available")
    
    try:
        voices = []
        tts_engine = "unknown"
        
        # Get TTS engine from config
        if config_manager:
            tts_engine = config_manager.config.speech.tts_engine
        
        # For pyttsx3, try to get actual voices
        if tts_engine == "pyttsx3" and hasattr(speech_processor, 'tts_engine'):
            try:
                if hasattr(speech_processor.tts_engine, 'engine'):
                    engine = speech_processor.tts_engine.engine
                    if hasattr(engine, 'getProperty'):
                        available_voices = engine.getProperty('voices')
                        if available_voices:
                            for i, v in enumerate(available_voices):
                                voices.append({
                                    "id": v.id if hasattr(v, 'id') else f"voice_{i}",
                                    "name": v.name if hasattr(v, 'name') else f"Voice {i}",
                                    "language": "en",  # Default assumption
                                    "gender": "unknown",
                                    "sample_rate": 16000
                                })
            except Exception as e:
                print(f"⚠️  Could not get pyttsx3 voices: {e}")
        
        # If no voices found, provide default options
        if not voices:
            voices = [
                {"id": "default", "name": "Default System Voice", "language": "en", "gender": "unknown", "sample_rate": 16000},
                {"id": "english", "name": "English", "language": "en", "gender": "unknown", "sample_rate": 16000},
                {"id": "english-us", "name": "English (US)", "language": "en-US", "gender": "unknown", "sample_rate": 16000},
                {"id": "english-uk", "name": "English (UK)", "language": "en-GB", "gender": "unknown", "sample_rate": 16000}
            ]
        
        # Get current voice from config
        current_voice = "default"
        if config_manager and hasattr(config_manager.config.speech, 'voice'):
            current_voice = config_manager.config.speech.voice
        
        return {
            "voices": voices,
            "count": len(voices),
            "tts_engine": tts_engine,
            "current_voice": current_voice,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing voices: {str(e)}")

@app.get("/models", tags=["AI"], response_model=Dict[str, Any])
async def list_available_models():
    """List available AI models"""
    try:
        # Define available models
        models = [
            {
                "id": "gpt-3.5-turbo",
                "name": "GPT-3.5 Turbo",
                "provider": "openai",
                "max_tokens": 16385,
                "context_window": 16385,
                "capabilities": ["chat", "completion", "function_calling"]
            },
            {
                "id": "gpt-4",
                "name": "GPT-4",
                "provider": "openai",
                "max_tokens": 8192,
                "context_window": 8192,
                "capabilities": ["chat", "completion", "function_calling", "reasoning"]
            },
            {
                "id": "gpt-4-turbo",
                "name": "GPT-4 Turbo",
                "provider": "openai",
                "max_tokens": 128000,
                "context_window": 128000,
                "capabilities": ["chat", "completion", "function_calling", "vision", "json_mode"]
            },
            {
                "id": "claude-3-haiku",
                "name": "Claude 3 Haiku",
                "provider": "anthropic",
                "max_tokens": 200000,
                "context_window": 200000,
                "capabilities": ["chat", "completion", "vision"]
            },
            {
                "id": "claude-3-sonnet",
                "name": "Claude 3 Sonnet",
                "provider": "anthropic",
                "max_tokens": 200000,
                "context_window": 200000,
                "capabilities": ["chat", "completion", "vision", "reasoning"]
            },
            {
                "id": "claude-3-opus",
                "name": "Claude 3 Opus",
                "provider": "anthropic",
                "max_tokens": 200000,
                "context_window": 200000,
                "capabilities": ["chat", "completion", "vision", "advanced_reasoning"]
            },
            {
                "id": "llama2",
                "name": "Llama 2",
                "provider": "local",
                "max_tokens": 4096,
                "context_window": 4096,
                "capabilities": ["chat", "completion"]
            },
            {
                "id": "mistral",
                "name": "Mistral",
                "provider": "local",
                "max_tokens": 32768,
                "context_window": 32768,
                "capabilities": ["chat", "completion"]
            },
            {
                "id": "gemini-pro",
                "name": "Gemini Pro",
                "provider": "google",
                "max_tokens": 32768,
                "context_window": 32768,
                "capabilities": ["chat", "completion", "vision"]
            }
        ]
        
        # Get current model from config
        current_model = "unknown"
        current_provider = "unknown"
        
        if config_manager and config_manager.config:
            current_model = config_manager.config.llm.model
            current_provider = config_manager.config.llm.provider
        
        # Check which providers are configured
        configured_providers = []
        if config_manager and config_manager.config:
            # Check if API keys are present (masked check)
            if hasattr(config_manager.config.llm, 'api_key') and config_manager.config.llm.api_key:
                configured_providers.append(config_manager.config.llm.provider)
        
        return {
            "models": models,
            "count": len(models),
            "current_model": current_model,
            "current_provider": current_provider,
            "configured_providers": configured_providers,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing models: {str(e)}")

@app.get("/system/info", tags=["Status"], response_model=Dict[str, Any])
async def system_info(): """Get detailed system information"""
    import platform
    import sys
    import psutil
    
    try:
        # System information
        system_info = {
            "python_version": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "system": platform.system(),
            "release": platform.release()
        }
        
        # Memory information
        memory = psutil.virtual_memory()
        memory_info = {
            "total_gb": memory.total / (1024**3),
            "available_gb": memory.available / (1024**3),
            "used_percent": memory.percent,
            "free_gb": memory.free / (1024**3)
        }
        
        # Disk information
        disk = psutil.disk_usage('/')
        disk_info = {
            "total_gb": disk.total / (1024**3),
            "used_gb": disk.used / (1024**3),
            "free_gb": disk.free / (1024**3),
            "used_percent": disk.percent
        }
        
        # Process information
        process = psutil.Process()
        process_info = {
            "pid": process.pid,
            "name": process.name(),
            "cpu_percent": process.cpu_percent(interval=0.1),
            "memory_mb": process.memory_info().rss / (1024**2),
            "status": process.status()
        }
        
        return {
            "system": system_info,
            "memory": memory_info,
            "disk": disk_info,
            "process": process_info,
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": (datetime.now() - startup_time).total_seconds()
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# ==================== ERROR HANDLERS ====================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "path": request.url.path,
            "method": request.method,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler"""
    # Log the full error
    print(f"❌ Unhandled exception: {exc}")
    print(traceback.format_exc())
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc),
            "path": request.url.path,
            "method": request.method,
            "timestamp": datetime.now().isoformat()
        }
    )

# ==================== WEBSOCKET ENDPOINT ====================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication"""
    await websocket.accept()
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            
            message_type = data.get("type", "message")
            
            if message_type == "query":
                # Process query
                text = data.get("text", "")
                if text and brain:
                    response = brain.process_query(text, data.get("context"))
                    
                    # Send response
                    await websocket.send_json({
                        "type": "response",
                        "text": response,
                        "timestamp": datetime.now().isoformat()
                    })
            
            elif message_type == "ping":
                # Respond to ping
                await websocket.send_json({
                    "type": "pong",
                    "timestamp": datetime.now().isoformat()
                })
            
            elif message_type == "disconnect":
                # Client wants to disconnect
                break
            
            else:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Unknown message type: {message_type}"
                })
                
    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "message": str(e)
            })
        except:
            pass

# ==================== STATIC FILES (for web interface) ====================

# Create static directory if it doesn't exist
static_dir = "static"
if not os.path.exists(static_dir):
    os.makedirs(static_dir)

# Create audio directory for generated audio files
audio_dir = os.path.join(static_dir, "audio")
if not os.path.exists(audio_dir):
    os.makedirs(audio_dir)

# Mount static files
app.mount("/static", StaticFiles(directory=static_dir), name="static")
app.mount("/audio", StaticFiles(directory=audio_dir), name="audio")

@app.get("/ui", tags=["Web Interface"], response_class=HTMLResponse)
async def serve_web_interface():
    """Serve the web interface"""
    ui_file = os.path.join(static_dir, "index.html")
    
    if not os.path.exists(ui_file):
        # Create a comprehensive web interface
        html_content = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Jarvis Assistant</title>
            <style>
                * {
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                }
                
                body {
                    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
                    color: #fff;
                    min-height: 100vh;
                    padding: 20px;
                }
                
                .container {
                    max-width: 1200px;
                    margin: 0 auto;
                    display: grid;
                    grid-template-columns: 1fr 2fr;
                    gap: 30px;
                }
                
                .header {
                    grid-column: 1 / -1;
                    text-align: center;
                    padding: 30px 0;
                    border-bottom: 2px solid rgba(255, 255, 255, 0.1);
                    margin-bottom: 30px;
                }
                
                .header h1 {
                    font-size: 3em;
                    background: linear-gradient(90deg, #00d2ff, #3a7bd5);
                    -webkit-background-clip: text;
                    background-clip: text;
                    color: transparent;
                    margin-bottom: 10px;
                }
                
                .header .tagline {
                    color: #a0a0a0;
                    font-size: 1.2em;
                }
                
                .sidebar {
                    background: rgba(255, 255, 255, 0.05);
                    border-radius: 15px;
                    padding: 25px;
                    backdrop-filter: blur(10px);
                    border: 1px solid rgba(255, 255, 255, 0.1);
                }
                
                .main-content {
                    background: rgba(255, 255, 255, 0.05);
                    border-radius: 15px;
                    padding: 25px;
                    backdrop-filter: blur(10px);
                    border: 1px solid rgba(255, 255, 255, 0.1);
                }
                
                h2 {
                    color: #00d2ff;
                    margin-bottom: 20px;
                    padding-bottom: 10px;
                    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
                }
                
                .status-card {
                    background: rgba(0, 210, 255, 0.1);
                    border: 1px solid rgba(0, 210, 255, 0.2);
                    border-radius: 10px;
                    padding: 15px;
                    margin-bottom: 20px;
                }
                
                .status-item {
                    display: flex;
                    justify-content: space-between;
                    margin-bottom: 10px;
                }
                
                .status-label {
                    color: #a0a0a0;
                }
                
                .status-value {
                    color: #00ff88;
                    font-weight: bold;
                }
                
                .control-panel {
                    margin-top: 30px;
                }
                
                .control-btn {
                    width: 100%;
                    padding: 12px;
                    margin-bottom: 10px;
                    background: rgba(0, 210, 255, 0.2);
                    border: 1px solid rgba(0, 210, 255, 0.3);
                    color: white;
                    border-radius: 8px;
                    cursor: pointer;
                    transition: all 0.3s;
                    font-size: 1em;
                }
                
                .control-btn:hover {
                    background: rgba(0, 210, 255, 0.3);
                    transform: translateY(-2px);
                }
                
                .chat-container {
                    height: 500px;
                    display: flex;
                    flex-direction: column;
                }
                
                .chat-messages {
                    flex: 1;
                    overflow-y: auto;
                    padding: 15px;
                    background: rgba(0, 0, 0, 0.2);
                    border-radius: 10px;
                    margin-bottom: 15px;
                }
                
                .message {
                    margin-bottom: 15px;
                    padding: 12px;
                    border-radius: 10px;
                    max-width: 80%;
                    word-wrap: break-word;
                }
                
                .user-message {
                    background: rgba(0, 123, 255, 0.2);
                    margin-left: auto;
                    border: 1px solid rgba(0, 123, 255, 0.3);
                }
                
                .jarvis-message {
                    background: rgba(40, 167, 69, 0.2);
                    margin-right: auto;
                    border: 1px solid rgba(40, 167, 69, 0.3);
                }
                
                .message-sender {
                    font-weight: bold;
                    margin-bottom: 5px;
                    font-size: 0.9em;
                }
                
                .message-content {
                    font-size: 1.1em;
                }
                
                .input-area {
                    display: flex;
                    gap: 10px;
                }
                
                #messageInput {
                    flex: 1;
                    padding: 12px;
                    border-radius: 8px;
                    border: 1px solid rgba(255, 255, 255, 0.2);
                    background: rgba(255, 255, 255, 0.05);
                    color: white;
                    font-size: 1em;
                }
                
                #sendButton {
                    padding: 12px 25px;
                    background: linear-gradient(90deg, #00d2ff, #3a7bd5);
                    border: none;
                    border-radius: 8px;
                    color: white;
                    cursor: pointer;
                    font-weight: bold;
                    transition: transform 0.2s;
                }
                
                #sendButton:hover {
                    transform: scale(1.05);
                }
                
                .audio-controls {
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 10px;
                    margin-top: 20px;
                }
                
                .audio-btn {
                    padding: 12px;
                    background: rgba(155, 89, 182, 0.2);
                    border: 1px solid rgba(155, 89, 182, 0.3);
                    color: white;
                    border-radius: 8px;
                    cursor: pointer;
                    transition: all 0.3s;
                }
                
                .audio-btn:hover {
                    background: rgba(155, 89, 182, 0.3);
                }
                
                @keyframes pulse {
                    0% { opacity: 1; }
                    50% { opacity: 0.5; }
                    100% { opacity: 1; }
                }
                
                .thinking {
                    animation: pulse 1.5s infinite;
                    color: #00d2ff;
                }
                
                .file-upload {
                    margin-top: 20px;
                    padding: 15px;
                    border: 2px dashed rgba(255, 255, 255, 0.2);
                    border-radius: 10px;
                    text-align: center;
                }
                
                .file-upload input {
                    display: none;
                }
                
                .file-upload label {
                    display: block;
                    padding: 10px;
                    background: rgba(0, 210, 255, 0.1);
                    border-radius: 8px;
                    cursor: pointer;
                }
                
                .tab-container {
                    margin-top: 20px;
                }
                
                .tabs {
                    display: flex;
                    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
                    margin-bottom: 20px;
                }
                
                .tab {
                    padding: 10px 20px;
                    cursor: pointer;
                    border-bottom: 3px solid transparent;
                }
                
                .tab.active {
                    border-bottom: 3px solid #00d2ff;
                    color: #00d2ff;
                }
                
                .tab-content {
                    display: none;
                }
                
                .tab-content.active {
                    display: block;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>JARVIS ASSISTANT</h1>
                    <p class="tagline">Your Personal AI Assistant - Always at Your Service</p>
                </div>
                
                <div class="sidebar">
                    <h2>System Status</h2>
                    <div class="status-card" id="statusCard">
                        <div class="status-item">
                            <span class="status-label">API Status:</span>
                            <span class="status-value" id="apiStatus">Checking...</span>
                        </div>
                        <div class="status-item">
                            <span class="status-label">AI Brain:</span>
                            <span class="status-value" id="aiStatus">Checking...</span>
                        </div>
                        <div class="status-item">
                            <span class="status-label">Speech Engine:</span>
                            <span class="status-value" id="speechStatus">Checking...</span>
                        </div>
                        <div class="status-item">
                            <span class="status-label">Response Time:</span>
                            <span class="status-value" id="responseTime">-- ms</span>
                        </div>
                        <div class="status-item">
                            <span class="status-label">Messages:</span>
                            <span class="status-value" id="messageCount">0</span>
                        </div>
                    </div>
                    
                    <div class="tab-container">
                        <div class="tabs">
                            <div class="tab active" onclick="switchTab('controls')">Controls</div>
                            <div class="tab" onclick="switchTab('audio')">Audio</div>
                            <div class="tab" onclick="switchTab('system')">System</div>
                        </div>
                        
                        <div id="controlsTab" class="tab-content active">
                            <div class="control-panel">
                                <h3>Quick Actions</h3>
                                <button class="control-btn" onclick="testConnection()">Test Connection</button>
                                <button class="control-btn" onclick="clearChat()">Clear Chat</button>
                                <button class="control-btn" onclick="testAudio()">Test Audio</button>
                                <button class="control-btn" onclick="getSystemInfo()">System Info</button>
                                <button class="control-btn" onclick="speak('Hello, I am Jarvis, your personal assistant. How can I help you today?')">
                                    Jarvis Intro
                                </button>
                            </div>
                        </div>
                        
                        <div id="audioTab" class="tab-content">
                            <div class="audio-controls">
                                <button class="audio-btn" onclick="startRecording()">🎤 Start Recording</button>
                                <button class="audio-btn" onclick="stopRecording()">⏹️ Stop Recording</button>
                                <button class="audio-btn" onclick="playLastAudio()">🔊 Play Last</button>
                                <button class="audio-btn" onclick="uploadAudio()">📁 Upload Audio</button>
                            </div>
                            
                            <div class="file-upload">
                                <h4>Upload Audio File</h4>
                                <input type="file" id="audioUpload" accept="audio/*" onchange="handleAudioUpload()">
                                <label for="audioUpload">Choose Audio File</label>
                                <p style="margin-top: 10px; font-size: 0.9em; color: #a0a0a0;">
                                    Supports: WAV, MP3, OGG, FLAC
                                </p>
                            </div>
                        </div>
                        
                        <div id="systemTab" class="tab-content">
                            <div class="control-panel">
                                <h3>System Commands</h3>
                                <button class="control-btn" onclick="getConfig()">Get Config</button>
                                <button class="control-btn" onclick="getModels()">List Models</button>
                                <button class="control-btn" onclick="getVoices()">List Voices</button>
                                <button class="control-btn" onclick="clearMemory()">Clear Memory</button>
                                <button class="control-btn" onclick="reconnectWebSocket()">Reconnect WS</button>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="main-content">
                    <h2>Chat with Jarvis</h2>
                    <div class="chat-container">
                        <div class="chat-messages" id="chatMessages">
                            <div class="message jarvis-message">
                                <div class="message-sender">Jarvis</div>
                                <div class="message-content">Hello! I'm Jarvis, your AI assistant. I'm ready to help you with questions, process audio, and assist with tasks. What can I do for you today?</div>
                            </div>
                        </div>
                        
                        <div class="input-area">
                            <input type="text" id="messageInput" placeholder="Type your message here..." onkeypress="handleKeyPress(event)">
                            <button id="sendButton" onclick="sendMessage()">Send</button>
                        </div>
                    </div>
                    
                    <div style="margin-top: 20px; color: #a0a0a0; font-size: 0.9em;">
                        <p>💡 <strong>Tips:</strong> Ask me anything, test audio functions, or check system status using the controls.</p>
                        <p>🔧 <strong>Commands:</strong> Try /help, /status, /clear, or /model info</p>
                    </div>
                </div>
            </div>

            <script>
                const API_BASE = 'http://localhost:8000';
                let messageCount = 1;
                let mediaRecorder;
                let audioChunks = [];
                let websocket = null;
                
                // Initialize
                document.addEventListener('DOMContentLoaded', function() {
                    updateStatus();
                    getSystemInfo();
                    connectWebSocket();
                });
                
                // Tab switching
                function switchTab(tabName) {
                    // Update tabs
                    document.querySelectorAll('.tab').forEach(tab => {
                        tab.classList.remove('active');
                    });
                    document.querySelectorAll('.tab-content').forEach(content => {
                        content.classList.remove('active');
                    });
                    
                    // Activate selected tab
                    event.target.classList.add('active');
                    document.getElementById(tabName + 'Tab').classList.add('active');
                }
                
                // Update system status
                async function updateStatus() {
                    try {
                        const response = await fetch(`${API_BASE}/health`);
                        const data = await response.json();
                        
                        document.getElementById('apiStatus').textContent = '✅ Online';
                        document.getElementById('aiStatus').textContent = data.components.brain ? '✅ Active' : '⚠️ Inactive';
                        document.getElementById('speechStatus').textContent = data.components.speech_processor ? '✅ Active' : '⚠️ Inactive';
                        
                        // Update status card color
                        const statusCard = document.getElementById('statusCard');
                        if (data.status === 'healthy') {
                            statusCard.style.background = 'rgba(0, 255, 135, 0.1)';
                            statusCard.style.borderColor = 'rgba(0, 255, 135, 0.2)';
                        } else {
                            statusCard.style.background = 'rgba(255, 193, 7, 0.1)';
                            statusCard.style.borderColor = 'rgba(255, 193, 7, 0.2)';
                        }
                        
                    } catch (error) {
                        document.getElementById('apiStatus').textContent = '❌ Offline';
                        document.getElementById('aiStatus').textContent = '❌ Unknown';
                        document.getElementById('speechStatus').textContent = '❌ Unknown';
                    }
                }
                
                // WebSocket connection
                function connectWebSocket() {
                    try {
                        websocket = new WebSocket(`ws://localhost:8000/ws`);
                        
                        websocket.onopen = function() {
                            console.log('WebSocket connected');
                        };
                        
                        websocket.onmessage = function(event) {
                            const data = JSON.parse(event.data);
                            if (data.type === 'response') {
                                addMessage('Jarvis', data.text, 'jarvis');
                            }
                        };
                        
                        websocket.onerror = function(error) {
                            console.error('WebSocket error:', error);
                        };
                        
                        websocket.onclose = function() {
                            console.log('WebSocket disconnected');
                            // Attempt to reconnect after 5 seconds
                            setTimeout(connectWebSocket, 5000);
                        };
                    } catch (error) {
                        console.error('Failed to connect WebSocket:', error);
                    }
                }
                
                // Send message to Jarvis
                async function sendMessage() {
                    const input = document.getElementById('messageInput');
                    const message = input.value.trim();
                    
                    if (!message) return;
                    
                    // Add user message to chat
                    addMessage('You', message, 'user');
                    input.value = '';
                    
                    // Show thinking indicator
                    const thinkingMsg = addMessage('Jarvis', 'Thinking...', 'jarvis');
                    thinkingMsg.classList.add('thinking');
                    
                    const startTime = Date.now();
                    
                    try {
                        const response = await fetch(`${API_BASE}/query`, {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({
                                text: message,
                                context: { interface: 'web', timestamp: new Date().toISOString() }
                            })
                        });
                        
                        const data = await response.json();
                        const endTime = Date.now();
                        
                        // Remove thinking message
                        thinkingMsg.remove();
                        
                        // Add Jarvis response
                        addMessage('Jarvis', data.response, 'jarvis');
                        
                        // Update metrics
                        document.getElementById('responseTime').textContent = `${endTime - startTime} ms`;
                        document.getElementById('messageCount').textContent = ++messageCount;
                        
                    } catch (error) {
                        thinkingMsg.remove();
                        addMessage('Jarvis', `Error: ${error.message}. Make sure the API server is running.`, 'jarvis');
                    }
                }
                
                // Add message to chat
                function addMessage(sender, content, type) {
                    const chatMessages = document.getElementById('chatMessages');
                    const messageDiv = document.createElement('div');
                    messageDiv.className = `message ${type}-message`;
                    
                    messageDiv.innerHTML = `
                        <div class="message-sender">${sender}</div>
                        <div class="message-content">${content}</div>
                    `;
                    
                    chatMessages.appendChild(messageDiv);
                    chatMessages.scrollTop = chatMessages.scrollHeight;
                    
                    return messageDiv;
                }
                
                // Handle Enter key
                function handleKeyPress(event) {
                    if (event.key === 'Enter') {
                        sendMessage();
                    }
                }
                
                // Test functions
                async function testConnection() {
                    addMessage('System', 'Testing connection to Jarvis API...', 'jarvis');
                    updateStatus();
                }
                
                function clearChat() {
                    document.getElementById('chatMessages').innerHTML = `
                        <div class="message jarvis-message">
                            <div class="message-sender">Jarvis</div>
                            <div class="message-content">Chat cleared. How can I assist you now?</div>
                        </div>
                    `;
                    messageCount = 1;
                    document.getElementById('messageCount').textContent = messageCount;
                }
                
                async function testAudio() {
                    try {
                        addMessage('System', 'Testing text-to-speech...', 'jarvis');
                        const response = await fetch(`${API_BASE}/speak?text=This is a test of the Jarvis audio system.`, {
                            method: 'POST'
                        });
                        const data = await response.json();
                        addMessage('System', `Audio test completed: ${data.message}`, 'jarvis');
                    } catch (error) {
                        addMessage('System', `Audio test failed: ${error.message}`, 'jarvis');
                    }
                }
                
                async function getSystemInfo() {
                    try {
                        const response = await fetch(`${API_BASE}/status`);
                        const data = await response.json();
                        
                        let info = `System Status:\\n`;
                        info += `• Status: ${data.status}\\n`;
                        info += `• Uptime: ${data.uptime_seconds?.toFixed(1)}s\\n`;
                        info += `• Brain: ${data.components?.brain?.status}\\n`;
                        info += `• Speech: ${data.components?.speech?.status}\\n`;
                        info += `• Model: ${data.components?.brain?.model}\\n`;
                        
                        addMessage('System', info, 'jarvis');
                    } catch (error) {
                        addMessage('System', `Could not fetch system info: ${error.message}`, 'jarvis');
                    }
                }
                
                // Audio recording functions
                async function startRecording() {
                    try {
                        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                        mediaRecorder = new MediaRecorder(stream);
                        audioChunks = [];
                        
                        mediaRecorder.ondataavailable = event => {
                            audioChunks.push(event.data);
                        };
                        
                        mediaRecorder.onstop = async () => {
                            const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                            
                            // Send to Jarvis for processing
                            addMessage('System', 'Processing recorded audio...', 'jarvis');
                            
                            const formData = new FormData();
                            formData.append('file', audioBlob, 'recording.wav');
                            
                            try {
                                const response = await fetch(`${API_BASE}/audio-query`, {
                                    method: 'POST',
                                    body: formData
                                });
                                
                                const data = await response.json();
                                addMessage('Jarvis', `Heard: "${data.text}"`, 'jarvis');
                                addMessage('Jarvis', data.response, 'jarvis');
                            } catch (error) {
                                addMessage('System', `Audio processing failed: ${error.message}`, 'jarvis');
                            }
                        };
                        
                        mediaRecorder.start();
                        addMessage('System', '🎤 Recording started... Speak now!', 'jarvis');
                        
                    } catch (error) {
                        addMessage('System', `Recording failed: ${error.message}`, 'jarvis');
                    }
                }
                
                function stopRecording() {
                    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
                        mediaRecorder.stop();
                        addMessage('System', '⏹️ Recording stopped. Processing...', 'jarvis');
                    }
                }
                
                function playLastAudio() {
                    // This would play the last generated audio
                    addMessage('System', 'Audio playback feature coming soon...', 'jarvis');
                }
                
                function uploadAudio() {
                    document.getElementById('audioUpload').click();
                }
                
                async function handleAudioUpload() {
                    const fileInput = document.getElementById('audioUpload');
                    const file = fileInput.files[0];
                    
                    if (!file) return;
                    
                    addMessage('System', `Uploading ${file.name}...`, 'jarvis');
                    
                    const formData = new FormData();
                    formData.append('file', file);
                    
                    try {
                        // First transcribe
                        const transcribeResponse = await fetch(`${API_BASE}/transcribe`, {
                            method: 'POST',
                            body: formData
                        });
                        
                        const transcribeData = await transcribeResponse.json();
                        addMessage('System', `Transcribed: "${transcribeData.text}"`, 'jarvis');
                        
                        // Then send to Jarvis
                        const queryResponse = await fetch(`${API_BASE}/query`, {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({
                                text: transcribeData.text,
                                context: { source: 'uploaded_audio' }
                            })
                        });
                        
                        const queryData = await queryResponse.json();
                        addMessage('Jarvis', queryData.response, 'jarvis');
                        
                    } catch (error) {
                        addMessage('System', `Upload failed: ${error.message}`, 'jarvis');
                    }
                    
                    // Reset file input
                    fileInput.value = '';
                }
                
                // Make text to speech
                async function speak(text) {
                    try {
                        await fetch(`${API_BASE}/speak?text=${encodeURIComponent(text)}`, {
                            method: 'POST'
                        });
                        addMessage('System', `Spoke: "${text}"`, 'jarvis');
                    } catch (error) {
                        addMessage('System', `Could not speak: ${error.message}`, 'jarvis');
                    }
                }
                
                // System functions
                async function getConfig() {
                    try {
                        const response = await fetch(`${API_BASE}/config`);
                        const data = await response.json();
                        addMessage('System', `Config loaded: ${JSON.stringify(data.config, null, 2)}`, 'jarvis');
                    } catch (error) {
                        addMessage('System', `Failed to get config: ${error.message}`, 'jarvis');
                    }
                }
                
                async function getModels() {
                    try {
                        const response = await fetch(`${API_BASE}/models`);
                        const data = await response.json();
                        
                        let modelsInfo = `Available Models (${data.count}):\\n`;
                        data.models.forEach(model => {
                            modelsInfo += `• ${model.name} (${model.provider})\\n`;
                        });
                        modelsInfo += `\\nCurrent: ${data.current_model} (${data.current_provider})`;
                        
                        addMessage('System', modelsInfo, 'jarvis');
                    } catch (error) {
                        addMessage('System', `Failed to get models: ${error.message}`, 'jarvis');
                    }
                }
                
                async function getVoices() {
                    try {
                        const response = await fetch(`${API_BASE}/voices`);
                        const data = await response.json();
                        
                        let voicesInfo = `Available Voices (${data.count}):\\n`;
                        data.voices.forEach(voice => {
                            voicesInfo += `• ${voice.name} (${voice.language})\\n`;
                        });
                        voicesInfo += `\\nCurrent: ${data.current_voice}`;
                        voicesInfo += `\\nEngine: ${data.tts_engine}`;
                        
                        addMessage('System', voicesInfo, 'jarvis');
                    } catch (error) {
                        addMessage('System', `Failed to get voices: ${error.message}`, 'jarvis');
                    }
                }
                
                async function clearMemory() {
                    try {
                        const response = await fetch(`${API_BASE}/memory/clear`, {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' }
                        });
                        const data = await response.json();
                        addMessage('System', data.message, 'jarvis');
                    } catch (error) {
                        addMessage('System', `Failed to clear memory: ${error.message}`, 'jarvis');
                    }
                }
                
                function reconnectWebSocket() {
                    if (websocket) {
                        websocket.close();
                    }
                    connectWebSocket();
                    addMessage('System', 'Reconnecting WebSocket...', 'jarvis');
                }
            </script>
        </body>
        </html>
        """
        
        with open(ui_file, 'w') as f:
            f.write(html_content)
    
    # Return the HTML file
    with open(ui_file, 'r') as f:
        return HTMLResponse(content=f.read())

# ==================== AUDIO FILE SERVING ====================

@app.get("/audio/list", tags=["Audio"])
async def list_audio_files():
    """List available audio files"""
    try:
        audio_files = []
        for file in os.listdir(audio_dir):
            if file.endswith(('.wav', '.mp3', '.ogg')):
                file_path = os.path.join(audio_dir, file)
                stat = os.stat(file_path)
                audio_files.append({
                    "filename": file,
                    "size": stat.st_size,
                    "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "url": f"/audio/{file}"
                })
        
        return {
            "audio_files": audio_files,
            "count": len(audio_files),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing audio files: {str(e)}")

@app.delete("/audio/{filename}", tags=["Audio"])
async def delete_audio_file(filename: str):
    """Delete an audio file"""
    try:
        # Security check: prevent path traversal
        if '..' in filename or '/' in filename or '\\' in filename:
            raise HTTPException(status_code=400, detail="Invalid filename")
        
        file_path = os.path.join(audio_dir, filename)
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        os.remove(file_path)
        
        return {
            "status": "success",
            "message": f"Deleted audio file: {filename}",
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting audio file: {str(e)}")

# ==================== ADDITIONAL UTILITY ENDPOINTS ====================

@app.get("/performance", tags=["Status"])
async def get_performance_stats():
    """Get performance statistics"""
    try:
        # Get brain performance stats if available
        brain_stats = {}
        if brain and hasattr(brain, 'get_performance_stats'):
            brain_stats = brain.get_performance_stats()
        
        # Calculate API uptime
        uptime = (datetime.now() - startup_time).total_seconds()
        
        # Get system info
        import psutil
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        return {
            "uptime_seconds": uptime,
            "brain_performance": brain_stats,
            "system": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3)
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting performance stats: {str(e)}")

@app.get("/conversation/summary", tags=["AI"])
async def get_conversation_summary():
    """Get conversation summary"""
    if not brain:
        raise HTTPException(status_code=503, detail="Brain component not available")
    
    try:
        if hasattr(brain, 'get_conversation_summary'):
            summary = brain.get_conversation_summary()
        else:
            # Fallback summary
            summary = {
                "total_exchanges": len(brain.conversation_history) // 2 if hasattr(brain, 'conversation_history') else 0,
                "memory_enabled": True
            }
        
        return {
            "summary": summary,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting conversation summary: {str(e)}")

# ==================== MAIN ENTRY POINT ====================

if __name__ == "__main__":
    uvicorn.run(
        "server.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True
    )