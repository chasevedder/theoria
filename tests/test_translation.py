"""Tests for validate_segments, parse_response, and translate_with_gemini."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from theoria.translation import validate_segments, parse_response


class TestValidateSegments:
    def test_valid_segments_pass_unchanged(self):
        segments = [
            {"start": 0.0, "end": 3.0, "english_text": "Hello", "text": "안녕"},
            {"start": 3.0, "end": 6.0, "english_text": "Bye", "text": "잘가"},
        ]
        result, skipped = validate_segments(segments)
        assert len(result) == 2
        assert skipped == 0

    def test_missing_numeric_start_end_skipped(self):
        segments = [
            {"start": "bad", "end": 3.0, "english_text": "Hello"},
            {"end": 3.0, "english_text": "No start"},
            {"start": 1.0, "english_text": "No end"},
        ]
        result, skipped = validate_segments(segments)
        assert len(result) == 0
        assert skipped == 3

    def test_text_fallback_to_english_text(self):
        segments = [{"start": 0.0, "end": 3.0, "text": "원본 텍스트"}]
        result, skipped = validate_segments(segments)
        assert len(result) == 1
        assert skipped == 0
        assert result[0]["english_text"] == "원본 텍스트"

    def test_malformed_captions_filtered(self):
        segments = [
            {
                "start": 0.0,
                "end": 3.0,
                "english_text": "Hello",
                "on_screen_captions": [
                    {"text": "Valid caption", "location": "top-center", "importance": "high"},
                    "just a string",
                    {"no_text_key": True},
                    {"text": "Also valid", "location": "bottom-center", "importance": "low"},
                ],
            }
        ]
        result, skipped = validate_segments(segments)
        assert len(result) == 1
        assert skipped == 0
        caps = result[0]["on_screen_captions"]
        assert len(caps) == 2
        assert caps[0]["text"] == "Valid caption"
        assert caps[1]["text"] == "Also valid"

    def test_empty_input(self):
        result, skipped = validate_segments([])
        assert result == []
        assert skipped == 0

    def test_mixed_valid_invalid(self):
        segments = [
            {"start": 0.0, "end": 2.0, "english_text": "Good"},
            {"start": "bad", "end": "bad"},
            {"start": 5.0, "end": 8.0, "english_text": "Also good"},
        ]
        result, skipped = validate_segments(segments)
        assert len(result) == 2
        assert skipped == 1
        assert result[0]["english_text"] == "Good"
        assert result[1]["english_text"] == "Also good"


class TestParseResponse:
    def _make_json(self, segments):
        return json.dumps(segments)

    def test_clean_json(self):
        text = self._make_json([{"start": 0.0, "end": 1.0, "english_text": "Hi"}])
        result = parse_response(text)
        assert len(result) == 1
        assert result[0]["english_text"] == "Hi"

    def test_strips_json_fence(self):
        inner = self._make_json([{"start": 0.0, "end": 1.0, "english_text": "Hi"}])
        text = f"```json\n{inner}\n```"
        result = parse_response(text)
        assert len(result) == 1

    def test_strips_plain_fence(self):
        inner = self._make_json([{"start": 0.0, "end": 1.0, "english_text": "Hi"}])
        text = f"```\n{inner}\n```"
        result = parse_response(text)
        assert len(result) == 1

    def test_empty_response_returns_none(self):
        assert parse_response(None) is None
        assert parse_response("") is None
        assert parse_response("   ") is None

    def test_invalid_json_returns_none(self):
        assert parse_response("not json at all") is None

    def test_validates_segments_in_response(self):
        text = self._make_json([
            {"start": 0.0, "end": 1.0, "english_text": "Good"},
            {"start": "bad", "end": "bad"},
        ])
        result = parse_response(text)
        assert len(result) == 1
        assert result[0]["english_text"] == "Good"


class TestTranslateWithGeminiFramesDir:
    """Verify frames_dir path construction for the frame_hash parameter."""

    _GOOGLE_MOCKS = {
        "google": MagicMock(),
        "google.genai": MagicMock(),
        "google.genai.types": MagicMock(),
    }

    def _make_mock_client(self, response_data):
        segments_json = json.dumps(response_data)
        mock_response = MagicMock()
        mock_response.text = segments_json
        mock_response.usage_metadata = MagicMock()
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        return mock_client

    def test_frames_dir_uses_frame_hash_when_provided(self, tmp_path):
        """translate_with_gemini reads frames from frames_{id}_{hash}/ when frame_hash set."""
        import sys
        from theoria.translation import translate_with_gemini
        from theoria.config import TheoriaConfig

        chunk = [{"start": 0.0, "end": 2.0, "speaker": "A", "text": "hi"}]
        response = [{"start": 0.0, "end": 2.0, "speaker": "A", "english_text": "hi"}]
        client = self._make_mock_client(response)

        frames_dir = tmp_path / "frames_0_abc12345"
        frames_dir.mkdir()
        (frames_dir / "chunk_0.jpg").write_bytes(b"")

        with patch.dict(sys.modules, self._GOOGLE_MOCKS):
            with patch("theoria.translation.Image.open", return_value=MagicMock()):
                result, _ = translate_with_gemini(
                    client, "video.mkv", chunk,
                    chunk_id=0, run_dir=tmp_path,
                    frame_hash="abc12345",
                    config=TheoriaConfig(),
                )

        assert result is not None

    def test_frames_dir_fallback_without_frame_hash(self, tmp_path):
        """translate_with_gemini falls back to frames_{id}/ when frame_hash is empty."""
        import sys
        from theoria.translation import translate_with_gemini
        from theoria.config import TheoriaConfig

        chunk = [{"start": 0.0, "end": 2.0, "speaker": "A", "text": "hi"}]
        response = [{"start": 0.0, "end": 2.0, "speaker": "A", "english_text": "hi"}]
        client = self._make_mock_client(response)

        frames_dir = tmp_path / "frames_0"
        frames_dir.mkdir()
        (frames_dir / "chunk_0.jpg").write_bytes(b"")

        with patch.dict(sys.modules, self._GOOGLE_MOCKS):
            with patch("theoria.translation.Image.open", return_value=MagicMock()):
                result, _ = translate_with_gemini(
                    client, "video.mkv", chunk,
                    chunk_id=0, run_dir=tmp_path,
                    frame_hash="",
                    config=TheoriaConfig(),
                )

        assert result is not None
