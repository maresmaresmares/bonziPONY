"""
Voice enrollment script — record your voice so Dash only listens to YOU.

Run this once:
    python scripts/enroll_voice.py

It will record ~8 seconds of you speaking, then save your voice profile.
After this, the transcriber will ignore YouTube, other people, TV, etc.
"""

import sys
import time
import struct
from pathlib import Path

import numpy as np
import pyaudio

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

SAMPLE_RATE = 16000
CHANNELS = 1
FRAME_DURATION_MS = 30
RECORD_SECONDS = 8


def main():
    print("=" * 60)
    print("  VOICE ENROLLMENT")
    print("  Speak naturally for ~8 seconds so Dash learns your voice.")
    print("  Talk about anything — just keep talking until it stops.")
    print("=" * 60)

    # Check for device index arg
    device_index = None
    if len(sys.argv) > 1:
        try:
            device_index = int(sys.argv[1])
            print(f"\nUsing audio device index: {device_index}")
        except ValueError:
            pass

    pa = pyaudio.PyAudio()
    frame_size = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)

    stream_kwargs = dict(
        format=pyaudio.paInt16,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=frame_size,
    )
    if device_index is not None:
        stream_kwargs["input_device_index"] = device_index

    input("\nPress ENTER when you're ready to start recording...")
    print("\n>>> RECORDING — speak now! <<<\n")

    stream = pa.open(**stream_kwargs)
    frames = []
    total_frames = int(RECORD_SECONDS * SAMPLE_RATE / frame_size)

    for i in range(total_frames):
        raw = stream.read(frame_size, exception_on_overflow=False)
        frames.append(raw)
        # Progress bar
        progress = (i + 1) / total_frames
        bar = "#" * int(progress * 40)
        remaining = RECORD_SECONDS - (i + 1) * FRAME_DURATION_MS / 1000
        print(f"\r  [{bar:<40}] {remaining:.1f}s remaining", end="", flush=True)

    stream.stop_stream()
    stream.close()
    pa.terminate()

    print("\n\n>>> Recording complete! Processing...\n")

    # Convert to float32
    audio_bytes = b"".join(frames)
    audio_int16 = struct.unpack(f"{len(audio_bytes) // 2}h", audio_bytes)
    audio_f32 = np.array(audio_int16, dtype=np.float32) / 32768.0

    # Enroll
    from stt.voice_filter import VoiceFilter
    vf = VoiceFilter()
    success = vf.enroll(audio_f32)

    if success:
        print("Voice profile saved! Dash will now only respond to YOUR voice.")
        print("YouTube, TV, and other people's speech will be ignored.")
        print("\nTo re-enroll, just run this script again.")
        print("To delete your profile, delete the voice_profile/ folder.")
    else:
        print("Enrollment failed. Make sure you spoke clearly during recording.")
        sys.exit(1)


if __name__ == "__main__":
    main()
