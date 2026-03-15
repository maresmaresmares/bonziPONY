"""
Stage-by-stage pipeline tester.

Usage:
    python scripts/test_pipeline.py stt        # speak and see transcript
    python scripts/test_pipeline.py llm        # type a message, see Dash's response
    python scripts/test_pipeline.py tts        # hear ElevenLabs output
    python scripts/test_pipeline.py robot      # print stub robot action
    python scripts/test_pipeline.py ack        # hear acknowledgement sound
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))


def _load_config():
    from core.config_loader import load_config
    return load_config(Path(__file__).parent.parent / "config.yaml")


def test_stt():
    print("=== STT Test ===")
    print("Speak now. Recording will stop after silence…")
    cfg = _load_config()
    from stt.transcriber import Transcriber
    t = Transcriber(
        model_name=cfg.whisper.model,
        language=cfg.whisper.language,
        vad_aggressiveness=cfg.audio.vad_aggressiveness,
        silence_duration_ms=cfg.audio.silence_duration_ms,
        input_device_index=cfg.audio.input_device_index,
    )
    result = t.listen()
    print(f"\nTranscription: {result!r}")


def test_llm():
    print("=== LLM Test ===")
    cfg = _load_config()
    from llm.factory import get_provider
    from llm.response_parser import parse_response

    provider = get_provider(cfg)
    while True:
        try:
            msg = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not msg:
            continue
        if msg.lower() in ("quit", "exit", "q"):
            break

        raw = provider.chat(msg)
        parsed = parse_response(raw)
        print(f"\nDash: {parsed.text}")
        if parsed.actions:
            print(f"[Actions: {[a.name for a in parsed.actions]}]")


def test_tts():
    print("=== TTS Test ===")
    cfg = _load_config()
    from tts.elevenlabs_tts import ElevenLabsTTS

    tts = ElevenLabsTTS(
        api_key=cfg.elevenlabs.api_key,
        voice_id=cfg.elevenlabs.voice_id,
        model=cfg.elevenlabs.model,
        output_format=cfg.elevenlabs.output_format,
        output_device_index=cfg.audio.output_device_index,
    )
    text = input("Text to speak (or Enter for default): ").strip()
    if not text:
        text = "Yeah! Twenty percent cooler, just like that! Totally awesome!"
    tts.speak(text)
    print("Done.")


def test_robot():
    print("=== Robot Stub Test ===")
    cfg = _load_config()
    from robot.unitree_stub import get_controller
    from robot.actions import RobotAction

    controller = get_controller(cfg)
    for action in RobotAction:
        controller.execute(action)
    print("All actions executed via stub.")


def test_ack():
    print("=== Acknowledgement Sound Test ===")
    from acknowledgement.player import AcknowledgementPlayer
    player = AcknowledgementPlayer()
    if not player._wavs:
        print("No .wav files in acknowledgement/assets/ — add some first.")
        return
    for _ in range(3):
        player.play()
    print("Done.")


TESTS = {
    "stt": test_stt,
    "llm": test_llm,
    "tts": test_tts,
    "robot": test_robot,
    "ack": test_ack,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in TESTS:
        print(__doc__)
        print(f"Available stages: {', '.join(TESTS)}")
        sys.exit(1)

    TESTS[sys.argv[1]]()
