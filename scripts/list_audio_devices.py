"""
Print all audio devices found by PyAudio, pvrecorder, and sounddevice.

Usage:
    python scripts/list_audio_devices.py

Use the printed indices to set audio.input_device_index and
audio.output_device_index in config.yaml.
"""

import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.audio_utils import list_pyaudio_devices, list_pvrecorder_devices, list_sounddevice_devices

if __name__ == "__main__":
    print("\n=== Audio Device Enumeration ===")
    list_sounddevice_devices()
    list_pvrecorder_devices()
    list_pyaudio_devices()
    print("\nDone. Set your preferred indices in config.yaml under 'audio:'.")
