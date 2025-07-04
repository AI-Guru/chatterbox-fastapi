from pydantic import BaseModel, Field, validator
from typing import Literal, Optional, Union


class TTSRequest(BaseModel):
    model: str = Field(
        default="tts-1", 
        description="The TTS model to use (currently ignored but required for compatibility)"
    )
    input: str = Field(
        description="The text to generate audio for", 
        max_length=4096
    )
    voice: Union[Literal["alloy", "echo", "fable", "nova", "onyx", "shimmer"], str] = Field(
        default="nova",
        description="The voice to use for audio generation (parameter kept for compatibility but ignored)"
    )
    response_format: Optional[Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]] = Field(
        default="mp3",
        description="The audio format for the response"
    )
    speed: Optional[float] = Field(
        default=1.0,
        ge=0.25,
        le=4.0,
        description="The speed of the generated audio (0.25 to 4.0)"
    )
    exaggeration: Optional[float] = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Emotion intensity and speech expressiveness (0.0-1.0)"
    )
    cfg: Optional[float] = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Classifier-free guidance for speech pacing (0.0-1.0)"
    )
    
    @validator("input")
    def validate_input_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Input text cannot be empty")
        return v


class TTSCloneRequest(BaseModel):
    model: str = Field(
        default="tts-1", 
        description="The TTS model to use (currently ignored but required for compatibility)"
    )
    input: str = Field(
        description="The text to generate audio for", 
        max_length=4096
    )
    voice: Union[Literal["alloy", "echo", "fable", "nova", "onyx", "shimmer"], str] = Field(
        default="custom",
        description="The voice to use for audio generation (parameter kept for compatibility)"
    )
    response_format: Optional[Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]] = Field(
        default="mp3",
        description="The audio format for the response"
    )
    speed: Optional[float] = Field(
        default=1.0,
        ge=0.25,
        le=4.0,
        description="The speed of the generated audio (0.25 to 4.0)"
    )
    exaggeration: Optional[float] = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Emotion intensity and speech expressiveness (0.0-1.0)"
    )
    cfg: Optional[float] = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Classifier-free guidance for speech pacing (0.0-1.0)"
    )
    
    @validator("input")
    def validate_input_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Input text cannot be empty")
        return v


class ErrorResponse(BaseModel):
    error: dict = Field(..., description="Error information")
    
    
class ErrorDetail(BaseModel):
    message: str = Field(..., description="Error message")
    type: str = Field(..., description="Error type")
    code: Optional[str] = Field(None, description="Error code")