from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uvicorn
import io
import tempfile
import os
import json
from datetime import datetime
import base64

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
try:
    config_manager = ConfigManager()
    brain = Brain(config_manager)
    speech_processor = SpeechProcessor(config_manager)
    print("✅ Jarvis components initialized successfully")
except Exception as e:
    print(f"⚠️  Warning: Component initialization failed: {e}")
    print("⚠️  Running in limited mode - some features may not work")
    config_manager = None
    brain = None
    speech_processor = None

# Request/Response Models
class QueryRequest(BaseModel):
    text: str
    context: Optional[dict] = None
    stream: Optional[bool] = False

class QueryResponse(BaseModel):
    query: str
    response: str
    timestamp: str
    processing_time: Optional[float] = None

class AudioResponse(BaseModel):
    text: str
    response: str
    timestamp: str
    audio_path: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str
    components: Dict[str, bool]

class AudioTranscriptionRequest(BaseModel):
    language: Optional[str] = "en"
    task: Optional[str] = "transcribe"

# ==================== HEALTH & STATUS ENDPOINTS ====================

@app.get("/", tags=["Status"])
async def root():
    """Root endpoint - API information"""
    return {
        "message": "Jarvis Assistant API",
        "status": "running",
        "version": "1.0.0",
        "endpoints": [
            "/docs - API documentation",
            "/health - System health check",
            "/query - Process text queries",
            "/transcribe - Transcribe audio",
            "/speak - Convert text to speech",
            "/audio-query - Full audio processing pipeline"
        ]
    }

@app.get("/health", response_model=HealthResponse, tags=["Status"])
async def health():
    """Health check endpoint"""
    components_status = {
        "config_manager": config_manager is not None,
        "brain": brain is not None,
        "speech_processor": speech_processor is not None,
        "fastapi": True,
        "uvicorn": True
    }
    
    status = "healthy" if all(components_status.values()) else "degraded"
    
    return HealthResponse(
        status=status,
        version="1.0.0",
        timestamp=datetime.now().isoformat(),
        components=components_status
    )

@app.get("/status", tags=["Status"])
async def status():
    """Detailed system status"""
    try:
        config = config_manager.config.dict() if config_manager else {}
        return {
            "status": "operational",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "brain": "active" if brain else "inactive",
                "speech": "active" if speech_processor else "inactive",
                "config": "loaded" if config_manager else "missing"
            },
            "endpoints_available": [
                "/health", "/query", "/transcribe", "/speak", "/audio-query", "/config"
            ]
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

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
        response = brain.process_query(request.text, request.context)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return QueryResponse(
            query=request.text,
            response=response,
            timestamp=datetime.now().isoformat(),
            processing_time=processing_time
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
    
    # This would be implemented with Server-Sent Events (SSE)
    # For now, return a placeholder
    return {"message": "Streaming endpoint - to be implemented"}

# ==================== SPEECH ENDPOINTS ====================

@app.post("/transcribe", tags=["Speech"])
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
        allowed_types = ["audio/wav", "audio/mpeg", "audio/mp3", "audio/ogg", "audio/flac"]
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Allowed: {', '.join(allowed_types)}"
            )
        
        # Read audio data
        audio_data = await file.read()
        
        # Create temporary file for processing
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_data)
            tmp_path = tmp.name
        
        try:
            # Transcribe using speech processor
            text = speech_processor.process_audio_file(tmp_path)
            
            return {
                "text": text,
                "filename": file.filename,
                "content_type": file.content_type,
                "size_bytes": len(audio_data),
                "timestamp": datetime.now().isoformat()
            }
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
                
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error transcribing audio: {str(e)}"
        )

@app.post("/speak", tags=["Speech"])
async def text_to_speech(
    text: str,
    background_tasks: BackgroundTasks,
    voice: Optional[str] = "default",
    speed: Optional[float] = 1.0
):
    """Convert text to speech and return audio file"""
    if not speech_processor:
        raise HTTPException(
            status_code=503,
            detail="Speech processor not available. Check initialization."
        )
    
    try:
        # Create temporary audio file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, delete_on_close=False) as tmp:
            tmp_path = tmp.name
        
        # Generate speech (you'll need to modify SpeechProcessor to support file output)
        # For now, using the existing speak_text method
        speech_processor.speak_text(text)
        
        # In a real implementation, you would:
        # 1. Generate audio file at tmp_path
        # 2. Return it with FileResponse
        # 3. Schedule cleanup with background_tasks
        
        return {
            "status": "success",
            "message": "Speech generated",
            "text": text,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating speech: {str(e)}"
        )

@app.post("/audio-query", response_model=AudioResponse, tags=["Speech", "AI"])
async def process_audio_query(file: UploadFile = File(...)):
    """Process audio query end-to-end: STT -> LLM -> TTS"""
    start_time = datetime.now()
    
    if not brain or not speech_processor:
        raise HTTPException(
            status_code=503,
            detail="Required components not available"
        )
    
    try:
        # Step 1: Transcribe audio
        audio_data = await file.read()
        query_text = speech_processor.stt_engine.transcribe(io.BytesIO(audio_data))
        
        # Step 2: Process with LLM
        response_text = brain.process_query(query_text)
        
        # Step 3: Generate speech
        speech_processor.speak_text(response_text)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return AudioResponse(
            text=query_text,
            response=response_text,
            timestamp=datetime.now().isoformat(),
            audio_path=None  # In future, return path to generated audio file
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing audio query: {str(e)}"
        )

# ==================== CONFIGURATION ENDPOINTS ====================

@app.get("/config", tags=["Configuration"])
async def get_config():
    """Get current configuration"""
    if not config_manager:
        raise HTTPException(status_code=503, detail="Config manager not available")
    
    try:
        return {
            "config": config_manager.config.dict(),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading config: {str(e)}")

@app.post("/config/update", tags=["Configuration"])
async def update_config(updates: dict):
    """Update configuration"""
    if not config_manager:
        raise HTTPException(status_code=503, detail="Config manager not available")
    
    try:
        config_manager.update_config(**updates)
        return {
            "status": "success",
            "message": "Configuration updated",
            "config": config_manager.config.dict(),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating config: {str(e)}")

@app.post("/memory/clear", tags=["AI"])
async def clear_memory():
    """Clear conversation memory"""
    if not brain:
        raise HTTPException(status_code=503, detail="Brain component not available")
    
    try:
        brain.clear_memory()
        return {
            "status": "success",
            "message": "Conversation memory cleared",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing memory: {str(e)}")

# ==================== UTILITY ENDPOINTS ====================

@app.get("/voices", tags=["Speech"])
async def list_available_voices():
    """List available TTS voices"""
    if not speech_processor:
        raise HTTPException(status_code=503, detail="Speech processor not available")
    
    try:
        # This would depend on your TTS engine implementation
        # For pyttsx3, you could do:
        # voices = speech_processor.tts_engine.engine.getProperty('voices')
        # return {"voices": [{"id": v.id, "name": v.name} for v in voices]}
        
        return {
            "voices": [
                {"id": "default", "name": "Default System Voice"},
                {"id": "english", "name": "English"},
                {"id": "english-us", "name": "English (US)"}
            ],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing voices: {str(e)}")

@app.get("/models", tags=["AI"])
async def list_available_models():
    """List available AI models"""
    if not brain:
        raise HTTPException(status_code=503, detail="Brain component not available")
    
    try:
        # This would depend on your brain implementation
        return {
            "models": [
                {"id": "gpt-3.5-turbo", "name": "GPT-3.5 Turbo", "provider": "openai"},
                {"id": "gpt-4", "name": "GPT-4", "provider": "openai"},
                {"id": "claude-3-haiku", "name": "Claude 3 Haiku", "provider": "anthropic"},
                {"id": "llama2", "name": "Llama 2", "provider": "local"}
            ],
            "current_model": getattr(brain, 'current_model', 'unknown'),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing models: {str(e)}")

# ==================== ERROR HANDLERS ====================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "path": request.url.path,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc),
            "path": request.url.path,
            "timestamp": datetime.now().isoformat()
        }
    )

# ==================== STARTUP/SHUTDOWN EVENTS ====================

@app.on_event("startup")
async def startup_event():
    """Run on application startup"""
    print(f"🚀 Jarvis Assistant API starting up at {datetime.now()}")
    print(f"📡 API Documentation available at http://localhost:8000/docs")

@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown"""
    print(f"🛑 Jarvis Assistant API shutting down at {datetime.now()}")
    # Cleanup code here if needed

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