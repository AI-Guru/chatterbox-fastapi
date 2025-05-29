# Chatterbox FastAPI Server

![Banner](banner.png)

OpenAI-compatible Text-to-Speech API server using the Chatterbox model by Resemble AI.

## Features

- 🎯 **OpenAI API Compatible**: Drop-in replacement for OpenAI's TTS API
- 🎤 **Voice Cloning**: Clone voices with 5-30 second audio samples  
- 🎵 **Multiple Formats**: MP3, WAV, FLAC, OPUS, AAC, PCM
- ⚡ **Speed Control**: 0.25x to 4.0x speed adjustment
- 🚀 **GPU Accelerated**: Requires CUDA-capable GPU

## Installation

**Requirements:** CUDA-capable GPU, Docker, Docker Compose

```bash
git clone https://github.com/yourusername/chatterbox-fastapi.git
cd chatterbox-fastapi
docker compose up
```

Server available at: `http://localhost:8308`

## API Endpoints

### POST `/v1/audio/speech`
Generate speech from text using form parameters.

### POST `/v1/audio/speech/clone` 
Generate speech with voice cloning by uploading an audio file.

### GET `/v1/models`
List available models (OpenAI compatible).

## Documentation

- **Swagger UI**: `http://localhost:8308/docs`
- **Health Check**: `http://localhost:8308/health`

## Example Usage

### OpenAI Python Client

```python
from openai import OpenAI

client = OpenAI(
    api_key="dummy-key", # We do not use API keys, but OpenAI client requires it.
    base_url="http://localhost:8308/v1"
)

response = client.audio.speech.create(
    model="tts-1",
    voice="nova",
    input="Hello from Chatterbox!"
)

response.stream_to_file("output.mp3")
```

## License

MIT License. [Chatterbox model](https://huggingface.co/ResembleAI/chatterbox) by Resemble AI.