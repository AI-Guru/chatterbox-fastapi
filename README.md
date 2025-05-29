# Chatterbox FastAPI Server

![Banner](banner.png)

An OpenAI-compatible Text-to-Speech (TTS) API server built with FastAPI and the Chatterbox model by Resemble AI. This server provides a drop-in replacement for OpenAI's TTS API endpoint, allowing you to use the open-source Chatterbox model with any application that supports the OpenAI TTS API.

## Features

- 🎯 **OpenAI API Compatible**: Implements the `/v1/audio/speech` endpoint with the same request/response format
- 🎙️ **Multiple Voices**: Supports 6 different voice profiles (alloy, echo, fable, nova, onyx, shimmer)
- 🎤 **Voice Cloning**: Clone any voice with just 5-30 seconds of audio sample
- 🎵 **Multiple Audio Formats**: Supports MP3, WAV, FLAC, OPUS, AAC, and PCM output formats
- ⚡ **Speed Control**: Adjustable speech speed from 0.25x to 4.0x
- 📚 **Auto-generated Documentation**: Interactive API documentation via Swagger UI
- 🐳 **Docker Ready**: Easy deployment with Docker and docker-compose
- 🔧 **Configurable**: Environment-based configuration for easy customization
- 🚀 **GPU Accelerated**: Requires CUDA-capable GPU for optimal performance

## Requirements

- Python 3.10+
- **CUDA-capable GPU (required)** - This application is GPU-only
- Docker and Docker Compose (for containerized deployment)

## Installation

### Local Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/chatterbox-fastapi.git
cd chatterbox-fastapi
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Copy the example environment file and configure it:
```bash
cp .env.example .env
```

5. Run the server:
```bash
python -m app.main
```

The server will start on `http://localhost:8000`

### Docker Deployment

1. Clone the repository:
```bash
git clone https://github.com/yourusername/chatterbox-fastapi.git
cd chatterbox-fastapi
```

2. Start the server using docker-compose:
```bash
docker compose up
```

The server will be available at `http://localhost:8000`

To run in detached mode:
```bash
docker compose up -d
```

## Configuration

The server can be configured using environment variables. Create a `.env` file based on `.env.example`:

```env
PORT=8000              # Server port
HOST=0.0.0.0          # Server host
DEVICE=cuda           # Device to use (cuda or cpu)
MODEL_CACHE_DIR=./models  # Directory to cache models
LOG_LEVEL=info        # Logging level
```

## API Usage

### Endpoint: POST `/v1/audio/speech`

Generate speech from text input.

#### Request Body

```json
{
  "model": "tts-1",
  "input": "Hello, this is a test of the Chatterbox TTS system.",
  "voice": "nova",
  "response_format": "mp3",
  "speed": 1.0
}
```

#### Parameters

- `model` (string, required): The TTS model to use (currently ignored but required for compatibility)
- `input` (string, required): The text to generate audio for (max 4096 characters)
- `voice` (string, optional): The voice to use. Options: `alloy`, `echo`, `fable`, `nova`, `onyx`, `shimmer`. Default: `nova`
- `response_format` (string, optional): The audio format. Options: `mp3`, `opus`, `aac`, `flac`, `wav`, `pcm`. Default: `mp3`
- `speed` (float, optional): The speed of the generated audio (0.25 to 4.0). Default: `1.0`

#### Example using cURL

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "Hello world!",
    "voice": "nova"
  }' \
  --output speech.mp3
```

#### Example using Python

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={
        "model": "tts-1",
        "input": "Hello from Python!",
        "voice": "echo",
        "response_format": "wav",
        "speed": 1.2
    }
)

with open("output.wav", "wb") as f:
    f.write(response.content)
```

#### Example using OpenAI Python Client

```python
from openai import OpenAI

# Point to your local server
client = OpenAI(
    api_key="dummy-key",  # API key is not used
    base_url="http://localhost:8000/v1"
)

response = client.audio.speech.create(
    model="tts-1",
    voice="nova",
    input="This works with the OpenAI client library!"
)

response.stream_to_file("output.mp3")
```

### Endpoint: POST `/v1/audio/speech/clone`

Generate speech with voice cloning from an audio sample.

#### Request Body

```json
{
  "model": "tts-1",
  "input": "Hello, this is a cloned voice speaking.",
  "voice": "custom",
  "audio_prompt_url": "https://example.com/voice-sample.wav",
  "response_format": "mp3",
  "speed": 1.0
}
```

#### Parameters

All parameters from the standard endpoint plus:
- `audio_prompt_url` (string, optional): URL to a WAV audio file for voice cloning (5-30 seconds recommended)

#### Example using cURL

```bash
curl -X POST http://localhost:8000/v1/audio/speech/clone \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "This is my cloned voice!",
    "audio_prompt_url": "https://example.com/my-voice.wav",
    "response_format": "wav"
  }' \
  --output cloned_speech.wav
```

#### Example using Python

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/audio/speech/clone",
    json={
        "model": "tts-1",
        "input": "Hello with a cloned voice!",
        "audio_prompt_url": "https://example.com/voice-sample.wav",
        "voice": "custom",
        "response_format": "mp3"
    }
)

with open("cloned_output.mp3", "wb") as f:
    f.write(response.content)
```

## API Documentation

Interactive API documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
- OpenAPI JSON: `http://localhost:8000/openapi.json`

## Voice Profiles

The server maps OpenAI's voice names to different Chatterbox model parameters:

- **alloy**: Clear and balanced
- **echo**: Slightly ethereal with more reverb
- **fable**: Expressive and dramatic
- **nova**: Natural and versatile (default)
- **onyx**: Deep and authoritative
- **shimmer**: Bright and energetic

## Performance Considerations

- **GPU Required**: This application requires a CUDA-capable GPU to run
- **Model Loading**: The first request may take longer as the model loads into memory
- **Batch Processing**: Currently processes one request at a time
- **Audio Format**: PCM format has the lowest overhead, while MP3/AAC require encoding

## Troubleshooting

### Common Issues

1. **CUDA not available**: This application requires a GPU with CUDA support. Ensure you have:
   - NVIDIA GPU with CUDA capability
   - Proper NVIDIA drivers installed
   - PyTorch with CUDA support installed

2. **Out of memory**: If you encounter GPU memory errors:
   - Close other GPU-intensive applications
   - Consider using a GPU with more VRAM
   - Reduce the maximum text length in requests

3. **Audio format not supported**: Some formats like AAC may fall back to MP3 depending on system codecs

4. **Model download fails**: Ensure you have internet connectivity and sufficient disk space

### Health Check

Check if the server is running properly:
```bash
curl http://localhost:8000/health
```

## Development

### Running Tests

```bash
pytest tests/
```

### Code Style

```bash
# Format code
black app/

# Lint code
flake8 app/
```

## License

This project is licensed under the MIT License. The Chatterbox model is also licensed under MIT by Resemble AI.

## Credits

- [Chatterbox](https://huggingface.co/ResembleAI/chatterbox) by Resemble AI
- Built with [FastAPI](https://fastapi.tiangolo.com/)
- OpenAI API specification for compatibility reference

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Disclaimer

This is an independent implementation and is not affiliated with OpenAI or Resemble AI.
