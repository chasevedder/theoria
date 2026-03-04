"""Tests for align_data with mock Whisper/Pyannote objects."""

from dataclasses import dataclass

from theoria.alignment import align_data


# ---- Lightweight mocks ----

@dataclass
class MockWord:
    word: str
    start: float
    end: float


@dataclass
class MockSegment:
    words: list[MockWord]


@dataclass
class MockTurn:
    start: float
    end: float


class MockDiarization:
    """Mimics pyannote Annotation.itertracks(yield_label=True)."""

    def __init__(self, turns: list[tuple[float, float, str]]):
        self._turns = turns

    def itertracks(self, yield_label=False):
        for start, end, speaker in self._turns:
            yield MockTurn(start, end), None, speaker


# ---- Tests ----

class TestAlignData:
    def test_basic_alignment(self):
        segments = [
            MockSegment(words=[
                MockWord(" Hello", 0.0, 0.5),
                MockWord(" world", 0.5, 1.0),
            ]),
        ]
        diarization = MockDiarization([(0.0, 2.0, "Speaker_A")])
        result = align_data(segments, diarization)
        speech = [s for s in result if s["text"]]
        assert len(speech) == 1
        assert speech[0]["start"] == 0.0
        assert speech[0]["end"] == 1.0
        assert speech[0]["speaker"] == "Speaker_A"
        assert speech[0]["text"] == "Hello world"

    def test_chunks_split_on_max_duration(self):
        words = [MockWord(f" w{i}", float(i), float(i + 1)) for i in range(6)]
        segments = [MockSegment(words=words)]
        diarization = MockDiarization([(0.0, 10.0, "Speaker_A")])
        result = align_data(segments, diarization, max_duration=3.0)
        speech = [s for s in result if s["text"]]
        assert len(speech) >= 2
        for seg in speech:
            assert seg["end"] - seg["start"] <= 3.01

    def test_gaps_filled_with_na(self):
        segments = [
            MockSegment(words=[MockWord(" Hello", 1.0, 2.0)]),
            MockSegment(words=[MockWord(" Bye", 5.0, 6.0)]),
        ]
        diarization = MockDiarization([(0.0, 2.0, "A"), (5.0, 6.0, "B")])
        result = align_data(segments, diarization, min_gap=0.5)
        na_segments = [s for s in result if s["speaker"] == "N/A"]
        assert len(na_segments) >= 1
        # Gap between 2.0 and 5.0 should be filled
        gap = [s for s in na_segments if s["start"] == 2.0 and s["end"] == 5.0]
        assert len(gap) == 1

    def test_gap_at_start_filled(self):
        segments = [MockSegment(words=[MockWord(" Hi", 2.0, 3.0)])]
        diarization = MockDiarization([(2.0, 3.0, "A")])
        result = align_data(segments, diarization, min_gap=0.5)
        # First entry should be a gap filler from 0.0 to 2.0
        assert result[0]["start"] == 0.0
        assert result[0]["end"] == 2.0
        assert result[0]["speaker"] == "N/A"

    def test_custom_min_gap_respected(self):
        segments = [
            MockSegment(words=[MockWord(" A", 0.0, 1.0)]),
            MockSegment(words=[MockWord(" B", 1.3, 2.0)]),
        ]
        diarization = MockDiarization([(0.0, 2.0, "X")])
        # Gap of 0.3s between segments — should NOT be filled with min_gap=0.5
        result = align_data(segments, diarization, min_gap=0.5)
        na_segments = [s for s in result if s["speaker"] == "N/A"]
        assert len(na_segments) == 0

    def test_custom_max_duration_respected(self):
        words = [MockWord(f" w{i}", float(i), float(i + 1)) for i in range(10)]
        segments = [MockSegment(words=words)]
        diarization = MockDiarization([(0.0, 20.0, "A")])
        result = align_data(segments, diarization, max_duration=2.0)
        speech = [s for s in result if s["text"]]
        for seg in speech:
            assert seg["end"] - seg["start"] <= 2.01
