"""Audio device enumeration helpers."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def list_pyaudio_devices() -> None:
    """Print all PyAudio input/output devices with indices."""
    try:
        import pyaudio
    except ImportError:
        print("PyAudio not installed. Run: pip install pyaudio")
        return

    pa = pyaudio.PyAudio()
    count = pa.get_device_count()
    print(f"\n{'='*60}")
    print(f"PyAudio devices ({count} total):")
    print(f"{'='*60}")
    for i in range(count):
        info = pa.get_device_info_by_index(i)
        tag = []
        if info["maxInputChannels"] > 0:
            tag.append("INPUT")
        if info["maxOutputChannels"] > 0:
            tag.append("OUTPUT")
        print(
            f"  [{i:2d}] {info['name']!r:40s}  "
            f"in={info['maxInputChannels']}  out={info['maxOutputChannels']}  "
            f"({', '.join(tag)})"
        )
    pa.terminate()


def list_pvrecorder_devices() -> None:
    """Print all pvrecorder-compatible input devices."""
    try:
        from pvrecorder import PvRecorder
    except ImportError:
        print("pvrecorder not installed. Run: pip install pvrecorder")
        return

    devices = PvRecorder.get_available_devices()
    print(f"\n{'='*60}")
    print(f"pvrecorder devices ({len(devices)} total):")
    print(f"{'='*60}")
    for i, name in enumerate(devices):
        print(f"  [{i:2d}] {name!r}")


def list_sounddevice_devices() -> None:
    """Print all sounddevice input/output devices."""
    try:
        import sounddevice as sd
    except ImportError:
        print("sounddevice not installed. Run: pip install sounddevice")
        return

    print(f"\n{'='*60}")
    print("sounddevice devices:")
    print(f"{'='*60}")
    print(sd.query_devices())
