import pytest

from veeksha.client.stt import TranscriptSnapshotRecorder


@pytest.mark.unit
def test_transcript_snapshot_recorder_dedupes_and_uses_audio_start() -> None:
    recorder = TranscriptSnapshotRecorder()

    recorder.mark_audio_started(101.0)
    recorder.add(100.5, "")
    recorder.add(101.25, "hello")
    recorder.add(101.30, "hello")
    recorder.add(101.50, "hello world")

    assert recorder.snapshots == [
        {"elapsed_ms": 250.0, "transcript": "hello"},
        {"elapsed_ms": 500.0, "transcript": "hello world"},
    ]
