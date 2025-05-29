from pydantic_settings import BaseSettings
from typing import Literal
import os


class Settings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8000
    device: Literal["cuda", "cpu"] = "cpu"  # Default to CPU for Docker compatibility
    model_cache_dir: str = "./models"
    log_level: str = "info"
    
    app_name: str = "Chatterbox FastAPI Server"
    app_version: str = "1.0.0"
    app_description: str = "OpenAI-compatible TTS API using Chatterbox model"
    
    max_text_length: int = 4096
    default_voice: str = "nova"
    default_speed: float = 1.0
    default_response_format: str = "mp3"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()