import torch
import torchaudio as ta
import numpy as np
from typing import Optional, Dict, Any
import io
import logging
import os
from chatterbox.tts import ChatterboxTTS
from app.config import settings

logger = logging.getLogger(__name__)

model_files = [
    "ve.pt",  # Voice encoder
    "t3_cfg.pt",  # Text-to-tokens model,
    "s3gen.pt",  # Speech generation model
    "conds.pt",  # Optional conditional embeddings
    "tokenizer.json",  # Tokenizer file
]



class TTSService:
    def __init__(self):
        self.model = None
        self.device = None
        self.sample_rate = None
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize the Chatterbox TTS model"""
        try:
            logger.info(f"Initializing Chatterbox TTS model on device: {settings.device}")
            
            # Check if CUDA is available
            if not torch.cuda.is_available():
                logger.error("CUDA is not available. This application requires a GPU.")
                raise RuntimeError("GPU is required to run this application. Please ensure CUDA is available.")
                
            device = settings.device
                
            self.device = device
            
            # Create model cache directory if it doesn't exist
            os.makedirs(settings.model_cache_dir, exist_ok=True)
            
            # Check if model is already cached locally
            cached_model_path = os.path.join(settings.model_cache_dir, "chatterbox")
            if not self._is_model_cached(cached_model_path):
                logger.info("Downloading model files to cache directory")
                self._download_model_to_cache(cached_model_path)
            
            logger.info(f"Loading model from cache: {cached_model_path}")
            self.model = ChatterboxTTS.from_local(cached_model_path, device=device)
                
            self.sample_rate = self.model.sr
            
            logger.info(f"Model initialized successfully on {device}")
            
        except Exception as e:
            logger.error(f"Failed to initialize TTS model: {str(e)}")
            raise
    
    def _is_model_cached(self, cache_path: str) -> bool:
        """Check if the model files are already cached locally"""
        
        # Check if all required files exist
        for file_name in model_files:
            file_path = os.path.join(cache_path, file_name)
            if not os.path.exists(file_path):
                return False
        
        logger.info(f"Found cached model files in {cache_path}")
        return True
    
    def _download_model_to_cache(self, cache_path: str):
        """Download model files directly to cache directory"""
        try:
            logger.info(f"Downloading model files to {cache_path}")
            os.makedirs(cache_path, exist_ok=True)
            
            from huggingface_hub import hf_hub_download
            import shutil
            
            # Download the required model files directly to cache
            repo_id = "ResembleAI/chatterbox"
            
            for file_name in model_files:
                logger.info(f"Downloading {file_name}...")
                # Download file and copy to our cache directory
                cache_file_path = os.path.join(cache_path, file_name)
                if os.path.exists(cache_file_path):
                    logger.info(f"{file_name} already exists in cache, skipping download")
                    continue
                hf_path = hf_hub_download(repo_id=repo_id, filename=file_name)
                shutil.copy2(hf_path, cache_file_path)
                logger.info(f"Downloaded {file_name} to {cache_file_path}")
            
            logger.info("Model download completed successfully")
                
        except Exception as e:
            logger.error(f"Failed to download model files: {str(e)}")
            raise
            
    def generate_speech(
        self,
        text: str,
        response_format: str = "wav",
        exaggeration: float = 0.5,
        cfg: float = 0.5,
        audio_prompt_path: Optional[str] = None
    ) -> bytes:
        """
        Generate speech from text using Chatterbox TTS
        
        Args:
            text: The text to convert to speech
            response_format: The output audio format
            exaggeration: Emotion intensity and speech expressiveness (0.0 to 1.0)
            cfg: Classifier-free guidance weight for speech pacing (0.0 to 1.0)
            audio_prompt_path: Path to audio file for voice cloning
            
        Returns:
            Audio data as bytes
        """
        try:
            # Voice parameter is kept for API compatibility but ignored
            logger.info(f"Generating speech for text: {text[:50]} with exaggeration: {exaggeration}, cfg: {cfg}, format: {response_format}")
            
            # Generate the audio waveform with specified parameters
            generate_kwargs = {
                "exaggeration": exaggeration,
                "cfg_weight": cfg,
            }
            
            # Add audio prompt for voice cloning if provided
            if audio_prompt_path:
                generate_kwargs["audio_prompt_path"] = audio_prompt_path
                logger.info(f"Using audio prompt for voice cloning: {audio_prompt_path}")
                
            # Generate the waveform.
            wav = self.model.generate(text, **generate_kwargs)
            
            # Convert to the requested format
            audio_bytes = self._convert_audio_format(wav, response_format)
            
            # Explicit cleanup of GPU tensors
            if torch.cuda.is_available():
                del wav
                torch.cuda.empty_cache()
            
            logger.info(f"Successfully generated {len(audio_bytes)} bytes of {response_format} audio")
            return audio_bytes
            
        except RuntimeError as e:
            error_msg = str(e)
            if "CUDA" in error_msg:
                logger.error(f"CUDA error during speech generation: {error_msg}")
                if "device-side assert" in error_msg:
                    raise RuntimeError("Invalid audio file for voice cloning. Please ensure the audio file is a valid WAV file with proper format (16kHz, mono recommended)")
                elif "out of memory" in error_msg:
                    raise RuntimeError("GPU out of memory. Try using a shorter audio sample or restart the service")
                else:
                    raise RuntimeError(f"GPU error during speech generation: {error_msg}")
            else:
                logger.error(f"Runtime error during speech generation: {error_msg}")
                raise
        except Exception as e:
            logger.error(f"Failed to generate speech: {str(e)}")
            raise
            
    
    def _convert_audio_format(self, wav: torch.Tensor, format: str) -> bytes:
        """Convert audio tensor to the requested format"""
        # Create a BytesIO buffer
        buffer = io.BytesIO()
        
        # Ensure wav is the right shape and type
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        
        if format == "wav":
            ta.save(buffer, wav, self.sample_rate, format="wav")
        elif format == "mp3":
            ta.save(buffer, wav, self.sample_rate, format="mp3")
        elif format == "flac":
            ta.save(buffer, wav, self.sample_rate, format="flac")
        elif format == "opus":
            # Opus encoding might require additional setup
            ta.save(buffer, wav, self.sample_rate, format="ogg", encoding="opus")
        elif format == "aac":
            # AAC might not be directly supported, fallback to mp3
            logger.warning("AAC format not directly supported, using MP3 instead")
            ta.save(buffer, wav, self.sample_rate, format="mp3")
        elif format == "pcm":
            # Raw PCM data
            pcm_data = (wav.numpy() * 32767).astype(np.int16)
            buffer.write(pcm_data.tobytes())
            del pcm_data  # Clean up numpy array
        else:
            # Default to wav
            ta.save(buffer, wav, self.sample_rate, format="wav")
            
        buffer.seek(0)
        audio_data = buffer.read()
        buffer.close()  # Explicit buffer cleanup
        return audio_data


# Global TTS service instance
tts_service = None


def get_tts_service() -> TTSService:
    """Get or create the global TTS service instance"""
    global tts_service
    if tts_service is None:
        tts_service = TTSService()
    return tts_service