from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import logging
from contextlib import asynccontextmanager
from app.config import settings
from app.models import TTSRequest, ErrorResponse, ErrorDetail
from app.tts_service import get_tts_service
import uvicorn
import tempfile
import aiofiles
import httpx
import os


# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    logger.info("Starting Chatterbox FastAPI server...")
    # Initialize TTS service on startup
    try:
        get_tts_service()
        logger.info("TTS service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize TTS service: {e}")
        raise
    yield
    logger.info("Shutting down Chatterbox FastAPI server...")


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    description=settings.app_description,
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "description": settings.app_description,
        "endpoints": {
            "tts": "/v1/audio/speech",
            "tts_clone": "/v1/audio/speech/clone",
            "models": "/v1/models",
            "docs": "/docs",
            "openapi": "/openapi.json"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check if TTS service is available
        tts = get_tts_service()
        if tts.model is None:
            raise Exception("TTS model not loaded")
        
        return {
            "status": "healthy",
            "service": "chatterbox-tts",
            "version": settings.app_version
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e)
            }
        )


@app.get("/v1/models")
async def list_models():
    """
    List available models (OpenAI API compatible)
    
    Returns a list of available TTS models for compatibility with OpenAI API clients.
    """
    return {
        "object": "list",
        "data": [
            {
                "id": "tts-1",
                "object": "model",
                "created": 1677610602,
                "owned_by": "resemble-ai",
                "permission": [],
                "root": "tts-1",
                "parent": None
            },
            {
                "id": "tts-1-hd",
                "object": "model", 
                "created": 1677610602,
                "owned_by": "resemble-ai",
                "permission": [],
                "root": "tts-1-hd",
                "parent": None
            }
        ]
    }


@app.post(
    "/v1/audio/speech",
    response_class=Response,
    responses={
        200: {
            "description": "Generated audio file",
            "content": {
                "audio/mpeg": {"schema": {"type": "string", "format": "binary"}},
                "audio/wav": {"schema": {"type": "string", "format": "binary"}},
                "audio/flac": {"schema": {"type": "string", "format": "binary"}},
                "audio/aac": {"schema": {"type": "string", "format": "binary"}},
                "audio/opus": {"schema": {"type": "string", "format": "binary"}},
                "audio/pcm": {"schema": {"type": "string", "format": "binary"}}
            }
        },
        400: {"model": ErrorResponse, "description": "Bad request"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    summary="Generate speech from text",
    description="Generate spoken audio from text input using the Chatterbox TTS model"
)
async def create_speech(
    model: str = Form(default="tts-1", description="The TTS model to use"),
    input: str = Form(default="To know that you know nothing is the first step to enlightenment.", description="The text to generate audio for"),
    voice: str = Form(default="nova", description="Voice parameter (kept for compatibility but ignored)"),
    response_format: str = Form(default="mp3", description="The audio format"),
    speed: float = Form(default=1.0, description="The speed of the generated audio")
):
    """
    OpenAI-compatible text-to-speech endpoint
    
    This endpoint accepts text input and returns generated audio in the specified format.
    It's designed to be compatible with OpenAI's TTS API.
    """
    try:
        logger.info(f"Received TTS request: voice={voice}, format={response_format}, speed={speed}")
        
        # Validate input text
        if not input or not input.strip():
            raise ValueError("Input text cannot be empty")
        
        # Validate speed
        if speed < 0.25 or speed > 4.0:
            raise ValueError("Speed must be between 0.25 and 4.0")
        
        # Validate response format
        valid_formats = ["mp3", "opus", "aac", "flac", "wav", "pcm"]
        if response_format not in valid_formats:
            raise ValueError(f"Invalid response format. Must be one of: {', '.join(valid_formats)}")
        
        # Get TTS service
        tts = get_tts_service()
        
        # Generate speech
        audio_data = tts.generate_speech(
            text=input,
            voice=voice,
            speed=speed,
            response_format=response_format
        )
        
        # Determine content type based on format
        content_types = {
            "mp3": "audio/mpeg",
            "opus": "audio/opus",
            "aac": "audio/aac",
            "flac": "audio/flac",
            "wav": "audio/wav",
            "pcm": "audio/pcm"
        }
        
        content_type = content_types.get(response_format, "audio/mpeg")
        
        return Response(
            content=audio_data,
            media_type=content_type,
            headers={
                "Content-Disposition": f'attachment; filename="speech.{response_format}"',
                "Content-Type": content_type
            }
        )
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": str(e),
                    "type": "invalid_request_error",
                    "code": "invalid_parameter"
                }
            }
        )
    except Exception as e:
        logger.error(f"Internal error during speech generation: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": "Internal server error during speech generation",
                    "type": "internal_error",
                    "code": "internal_error"
                }
            }
        )


@app.post(
    "/v1/audio/speech/clone",
    response_class=Response,
    responses={
        200: {
            "description": "Generated audio file with cloned voice",
            "content": {
                "audio/mpeg": {"schema": {"type": "string", "format": "binary"}},
                "audio/wav": {"schema": {"type": "string", "format": "binary"}},
                "audio/flac": {"schema": {"type": "string", "format": "binary"}},
                "audio/aac": {"schema": {"type": "string", "format": "binary"}},
                "audio/opus": {"schema": {"type": "string", "format": "binary"}},
                "audio/pcm": {"schema": {"type": "string", "format": "binary"}}
            }
        },
        400: {"model": ErrorResponse, "description": "Bad request"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    summary="Generate speech with voice cloning",
    description="Generate spoken audio from text input using voice cloning from an uploaded audio sample"
)
async def create_speech_with_cloning(
    model: str = Form(default="tts-1", description="The TTS model to use"),
    input: str = Form(default="To know that you know nothing is the first step to enlightenment.", description="The text to generate audio for"),
    voice: str = Form(default="custom", description="Voice parameter (kept for compatibility)"),
    response_format: str = Form(default="mp3", description="The audio format"),
    speed: float = Form(default=1.0, description="The speed of the generated audio"),
    audio_prompt: UploadFile = File(..., description="Audio file for voice cloning (WAV format recommended)")
):
    """
    Extended TTS endpoint with voice cloning support
    
    This endpoint accepts text input and an uploaded audio file to clone the voice.
    The audio sample should be a WAV file for best results.
    """
    audio_prompt_path = None
    temp_file = None
    
    try:
        logger.info(f"Received voice cloning TTS request: format={response_format}, speed={speed}")
        
        # Validate input text
        if not input or not input.strip():
            raise ValueError("Input text cannot be empty")
        
        # Validate speed
        if speed < 0.25 or speed > 4.0:
            raise ValueError("Speed must be between 0.25 and 4.0")
        
        # Validate response format
        valid_formats = ["mp3", "opus", "aac", "flac", "wav", "pcm"]
        if response_format not in valid_formats:
            raise ValueError(f"Invalid response format. Must be one of: {', '.join(valid_formats)}")
        
        # Save uploaded audio file to temporary location
        if audio_prompt:
            logger.info(f"Processing uploaded audio file: {audio_prompt.filename}")
            
            # Create temporary file for audio prompt
            file_extension = audio_prompt.filename.split('.')[-1] if '.' in audio_prompt.filename else 'wav'
            temp_file = tempfile.NamedTemporaryFile(suffix=f".{file_extension}", delete=False)
            audio_prompt_path = temp_file.name
            
            # Save uploaded file content
            content = await audio_prompt.read()
            async with aiofiles.open(audio_prompt_path, 'wb') as f:
                await f.write(content)
            
            temp_file.close()
            logger.info(f"Audio prompt saved to: {audio_prompt_path}")
        
        # Get TTS service
        tts = get_tts_service()
        
        # Generate speech with voice cloning
        audio_data = tts.generate_speech(
            text=input,
            voice="custom" if audio_prompt_path else voice,
            speed=speed,
            response_format=response_format,
            audio_prompt_path=audio_prompt_path
        )
        
        # Determine content type based on format
        content_types = {
            "mp3": "audio/mpeg",
            "opus": "audio/opus",
            "aac": "audio/aac",
            "flac": "audio/flac",
            "wav": "audio/wav",
            "pcm": "audio/pcm"
        }
        
        content_type = content_types.get(response_format, "audio/mpeg")
        
        return Response(
            content=audio_data,
            media_type=content_type,
            headers={
                "Content-Disposition": f'attachment; filename="speech_cloned.{response_format}"',
                "Content-Type": content_type
            }
        )
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": str(e),
                    "type": "invalid_request_error",
                    "code": "invalid_parameter"
                }
            }
        )
    except Exception as e:
        logger.error(f"Internal error during speech generation: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": "Internal server error during speech generation",
                    "type": "internal_error",
                    "code": "internal_error"
                }
            }
        )
    finally:
        # Clean up temporary file
        if temp_file and os.path.exists(audio_prompt_path):
            try:
                os.unlink(audio_prompt_path)
                logger.info(f"Cleaned up temporary file: {audio_prompt_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file: {e}")


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        log_level=settings.log_level
    )