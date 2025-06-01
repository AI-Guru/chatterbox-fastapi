#!/usr/bin/env python3
"""
Test script for Chatterbox FastAPI endpoints using the requests package.
Tests both /v1/audio/speech and /v1/audio/speech/clone endpoints.
"""

import requests
import json
import os
import tempfile
from pathlib import Path

BASE_URL = "http://localhost:8308"

def test_basic_speech_endpoint():
    """Test the basic /v1/audio/speech endpoint"""
    print("Testing /v1/audio/speech endpoint...")
    
    url = f"{BASE_URL}/v1/audio/speech"
    
    # Test data
    data = {
        "model": "tts-1",
        "input": "Hello from Chatterbox! This is a test of the speech synthesis API.",
        "voice": "nova",
        "response_format": "mp3",
        "speed": 1.0,
        "exaggeration": 0.7,
        "cfg": 0.6
    }
    
    try:
        response = requests.post(url, json=data, timeout=60)
        
        if response.status_code == 200:
            # Save the audio file
            output_file = "test_output_basic.mp3"
            with open(output_file, "wb") as f:
                f.write(response.content)
            
            print(f"✓ Basic speech test successful!")
            print(f"  - Status: {response.status_code}")
            print(f"  - Content-Type: {response.headers.get('content-type')}")
            print(f"  - Audio size: {len(response.content)} bytes")
            print(f"  - Saved to: {output_file}")
            return True
        else:
            print(f"✗ Basic speech test failed!")
            print(f"  - Status: {response.status_code}")
            print(f"  - Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"✗ Basic speech test error: {e}")
        return False

def test_speech_with_different_formats():
    """Test different audio formats"""
    print("\nTesting different audio formats...")
    
    url = f"{BASE_URL}/v1/audio/speech"
    formats = ["mp3", "wav", "flac"]
    
    results = []
    for fmt in formats:
        print(f"  Testing {fmt} format...")
        
        data = {
            "model": "tts-1",
            "input": f"Testing {fmt} format output.",
            "voice": "nova",
            "response_format": fmt,
            "speed": 1.0
        }
        
        try:
            response = requests.post(url, json=data, timeout=60)
            
            if response.status_code == 200:
                output_file = f"test_output_{fmt}.{fmt}"
                with open(output_file, "wb") as f:
                    f.write(response.content)
                
                print(f"    ✓ {fmt} format successful ({len(response.content)} bytes)")
                results.append(True)
            else:
                print(f"    ✗ {fmt} format failed: {response.status_code}")
                results.append(False)
                
        except Exception as e:
            print(f"    ✗ {fmt} format error: {e}")
            results.append(False)
    
    return all(results)

def test_speech_clone_endpoint():
    """Test the voice cloning endpoint with a sample audio file"""
    print("\nTesting /v1/audio/speech/clone endpoint...")
    
    url = f"{BASE_URL}/v1/audio/speech/clone"
    
    # Create a minimal but valid WAV file for testing
    # This creates a WAV file with silence (about 1 second at 16kHz)
    wav_header = b'RIFF'
    chunk_size = (16000 * 2 + 36).to_bytes(4, 'little')  # file size - 8
    wave_format = b'WAVE'
    fmt_chunk = b'fmt '
    fmt_chunk_size = (16).to_bytes(4, 'little')
    audio_format = (1).to_bytes(2, 'little')  # PCM
    num_channels = (1).to_bytes(2, 'little')  # mono
    sample_rate = (16000).to_bytes(4, 'little')  # 16kHz
    byte_rate = (16000 * 2).to_bytes(4, 'little')
    block_align = (2).to_bytes(2, 'little')
    bits_per_sample = (16).to_bytes(2, 'little')
    data_chunk = b'data'
    data_size = (16000 * 2).to_bytes(4, 'little')  # 1 second of 16-bit mono audio
    audio_data = b'\x00' * (16000 * 2)  # silence
    
    dummy_audio_content = (wav_header + chunk_size + wave_format + fmt_chunk + 
                          fmt_chunk_size + audio_format + num_channels + sample_rate + 
                          byte_rate + block_align + bits_per_sample + data_chunk + 
                          data_size + audio_data)
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_file.write(dummy_audio_content)
        temp_file_path = temp_file.name
    
    try:
        # Prepare form data
        files = {
            'audio_prompt': ('test.wav', open(temp_file_path, 'rb'), 'audio/wav')
        }
        
        data = {
            'model': 'tts-1',
            'input': 'This is a test of voice cloning functionality.',
            'voice': 'custom',
            'response_format': 'mp3',
            'speed': 1.0,
            'exaggeration': 0.5,
            'cfg': 0.5
        }
        
        response = requests.post(url, files=files, data=data, timeout=60)
        
        if response.status_code == 200:
            output_file = "test_output_cloned.mp3"
            with open(output_file, "wb") as f:
                f.write(response.content)
            
            print(f"✓ Voice cloning test successful!")
            print(f"  - Status: {response.status_code}")
            print(f"  - Content-Type: {response.headers.get('content-type')}")
            print(f"  - Audio size: {len(response.content)} bytes")
            print(f"  - Saved to: {output_file}")
            return True
        else:
            print(f"✗ Voice cloning test failed!")
            print(f"  - Status: {response.status_code}")
            print(f"  - Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"✗ Voice cloning test error: {e}")
        return False
    finally:
        # Clean up temporary file
        files['audio_prompt'][1].close()
        os.unlink(temp_file_path)

def test_validation_errors():
    """Test various validation scenarios"""
    print("\nTesting validation scenarios...")
    
    url = f"{BASE_URL}/v1/audio/speech"
    
    test_cases = [
        {
            "name": "Empty input",
            "data": {"model": "tts-1", "input": "", "voice": "nova"},
            "expected_status": 422
        },
        {
            "name": "Invalid speed (too high)",
            "data": {"model": "tts-1", "input": "Test", "voice": "nova", "speed": 5.0},
            "expected_status": 422
        },
        {
            "name": "Invalid speed (too low)",
            "data": {"model": "tts-1", "input": "Test", "voice": "nova", "speed": 0.1},
            "expected_status": 422
        },
        {
            "name": "Invalid exaggeration",
            "data": {"model": "tts-1", "input": "Test", "voice": "nova", "exaggeration": 2.0},
            "expected_status": 422
        },
        {
            "name": "Missing required input field",
            "data": {"model": "tts-1", "voice": "nova"},
            "expected_status": 422
        },
        {
            "name": "Invalid response format",
            "data": {"model": "tts-1", "input": "Test", "voice": "nova", "response_format": "invalid"},
            "expected_status": 422
        }
    ]
    
    results = []
    for case in test_cases:
        print(f"  Testing {case['name']}...")
        
        try:
            response = requests.post(url, json=case["data"], timeout=10)
            
            if response.status_code == case["expected_status"]:
                print(f"    ✓ Validation working correctly (status: {response.status_code})")
                results.append(True)
            else:
                print(f"    ✗ Unexpected status: {response.status_code} (expected: {case['expected_status']})")
                if case["name"] == "Missing required input field":
                    print(f"      Response: {response.text[:200]}")
                results.append(False)
                
        except Exception as e:
            print(f"    ✗ Error: {e}")
            results.append(False)
    
    return all(results)

def test_health_endpoint():
    """Test the health check endpoint"""
    print("\nTesting health endpoint...")
    
    url = f"{BASE_URL}/health"
    
    try:
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            health_data = response.json()
            print(f"✓ Health check successful!")
            print(f"  - Status: {health_data.get('status')}")
            print(f"  - Service: {health_data.get('service')}")
            print(f"  - Version: {health_data.get('version')}")
            return True
        else:
            print(f"✗ Health check failed: {response.status_code}")
            print(f"  - Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"✗ Health check error: {e}")
        return False

def main():
    """Run all tests"""
    print("🎤 Chatterbox FastAPI Endpoint Tests")
    print("=" * 50)
    
    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code not in [200, 503]:
            print("❌ Server doesn't seem to be running or accessible")
            print(f"Make sure the server is running on {BASE_URL}")
            return
    except requests.exceptions.RequestException:
        print("❌ Server doesn't seem to be running or accessible")
        print(f"Make sure the server is running on {BASE_URL}")
        return
    
    # Run tests
    tests = [
        ("Health Check", test_health_endpoint),
        ("Basic Speech Generation", test_basic_speech_endpoint),
        ("Different Audio Formats", test_speech_with_different_formats),
        ("Voice Cloning", test_speech_clone_endpoint),
        ("Validation Errors", test_validation_errors),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📝 Running: {test_name}")
        print("-" * 30)
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} - {test_name}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")

if __name__ == "__main__":
    main()