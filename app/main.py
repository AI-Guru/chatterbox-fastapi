from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import logging
from contextlib import asynccontextmanager
from app.config import settings
from app.models import TTSRequest, VoiceCloningRequest, ErrorResponse, ErrorDetail
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


@app.post(
    "/v1/audio/speech",
    response_class=Response,
    responses={
        200: {
            "description": "Generated audio file",
            "content": {
                "audio/mpeg": {},
                "audio/wav": {},
                "audio/flac": {},
                "audio/aac": {},
                "audio/opus": {},
                "audio/pcm": {}
            }
        },
        400: {"model": ErrorResponse, "description": "Bad request"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    summary="Generate speech from text",
    description="Generate spoken audio from text input using the Chatterbox TTS model"
)
async def create_speech(request: TTSRequest):
    """
    OpenAI-compatible text-to-speech endpoint
    
    This endpoint accepts text input and returns generated audio in the specified format.
    It's designed to be compatible with OpenAI's TTS API.
    """
    try:
        logger.info(f"Received TTS request: voice={request.voice}, format={request.response_format}, speed={request.speed}")
        
        # Get TTS service
        tts = get_tts_service()
        
        # Generate speech
        audio_data = tts.generate_speech(
            text=request.input,
            voice=request.voice,
            speed=request.speed,
            response_format=request.response_format
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
        
        content_type = content_types.get(request.response_format, "audio/mpeg")
        
        return Response(
            content=audio_data,
            media_type=content_type,
            headers={
                "Content-Disposition": f'inline; filename="speech.{request.response_format}"'
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
                "audio/mpeg": {},
                "audio/wav": {},
                "audio/flac": {},
                "audio/aac": {},
                "audio/opus": {},
                "audio/pcm": {}
            }
        },
        400: {"model": ErrorResponse, "description": "Bad request"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    summary="Generate speech with voice cloning",
    description="Generate spoken audio from text input using voice cloning from an audio sample"
)
async def create_speech_with_cloning(request: VoiceCloningRequest):
    """
    Extended TTS endpoint with voice cloning support
    
    This endpoint accepts text input and an audio sample URL to clone the voice.
    The audio sample should be a WAV file for best results.
    """
    audio_prompt_path = None
    temp_file = None
    
    try:
        logger.info(f"Received voice cloning TTS request: format={request.response_format}, speed={request.speed}")
        
        # Download audio prompt if provided
        if request.audio_prompt_url:
            logger.info(f"Downloading audio prompt from: {request.audio_prompt_url}")
            
            # Create temporary file for audio prompt
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            audio_prompt_path = temp_file.name
            temp_file.close()
            
            # Download the audio file
            async with httpx.AsyncClient() as client:
                response = await client.get(request.audio_prompt_url)
                response.raise_for_status()
                
                # Save to temporary file
                async with aiofiles.open(audio_prompt_path, 'wb') as f:
                    await f.write(response.content)
                    
            logger.info(f"Audio prompt downloaded to: {audio_prompt_path}")
        
        # Get TTS service
        tts = get_tts_service()
        
        # Generate speech with voice cloning
        audio_data = tts.generate_speech(
            text=request.input,
            voice="custom" if audio_prompt_path else request.voice,
            speed=request.speed,
            response_format=request.response_format,
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
        
        content_type = content_types.get(request.response_format, "audio/mpeg")
        
        return Response(
            content=audio_data,
            media_type=content_type,
            headers={
                "Content-Disposition": f'inline; filename="speech_cloned.{request.response_format}"'
            }
        )
        
    except httpx.HTTPError as e:
        logger.error(f"Error downloading audio prompt: {e}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": f"Failed to download audio prompt: {str(e)}",
                    "type": "invalid_request_error",
                    "code": "audio_download_failed"
                }
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