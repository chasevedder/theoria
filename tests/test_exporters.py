"""Tests for timestamp formatting and SRT/ASS export."""

from theoria.exporters import (
    format_timestamp,
    format_timestamp_ass,
    export_srt,
    export_ass,
)
from theoria.config import TheoriaConfig


class TestFormatTimestamp:
    def test_zero(self):
        assert format_timestamp(0.0) == "00:00:00,000"

    def test_fractional_seconds(self):
        assert format_timestamp(19.24) == "00:00:19,240"

    def test_over_one_hour(self):
        assert format_timestamp(3661.5) == "01:01:01,500"


class TestFormatTimestampAss:
    def test_zero(self):
        assert format_timestamp_ass(0.0) == "0:00:00.00"

    def test_fractional_seconds(self):
        assert format_timestamp_ass(19.24) == "0:00:19.24"

    def test_over_one_hour(self):
        assert format_timestamp_ass(3661.5) == "1:01:01.50"


class TestExportSrt:
    def test_writes_valid_srt(self, tmp_path, sample_segments):
        out = tmp_path / "test.srt"
        export_srt(sample_segments, str(out))
        content = out.read_text()
        # Should have numbered entries
        assert "1\n" in content
        assert "-->" in content

    def test_only_high_importance_captions(self, tmp_path, sample_segments):
        out = tmp_path / "test.srt"
        export_srt(sample_segments, str(out))
        content = out.read_text()
        assert "[First appearance!]" in content
        assert "[Location: Seoul]" in content
        # Low-importance "wow" should NOT appear
        assert "wow" not in content

    def test_skips_empty_dialogue_no_captions(self, tmp_path):
        segments = [{"start": 0.0, "end": 2.0, "speaker": "N/A", "text": "", "english_text": ""}]
        out = tmp_path / "test.srt"
        export_srt(segments, str(out))
        content = out.read_text()
        assert content.strip() == ""

    def test_captions_bracketed(self, tmp_path, sample_segments):
        out = tmp_path / "test.srt"
        export_srt(sample_segments, str(out))
        content = out.read_text()
        assert "[First appearance!]" in content

    def test_respects_max_cap_duration(self, tmp_path):
        segments = [
            {
                "start": 0.0,
                "end": 20.0,
                "speaker": "Host",
                "text": "",
                "english_text": "Hello",
                "on_screen_captions": [
                    {"text": "Cap", "location": "top-center", "importance": "high"},
                ],
            }
        ]
        cfg = TheoriaConfig(max_cap_duration=2.0)
        out = tmp_path / "test.srt"
        export_srt(segments, str(out), config=cfg)
        content = out.read_text()
        # Caption window should be capped at 2s around midpoint (10s)
        # So caption should start around 9.0 and end around 11.0
        assert "00:00:09,000" in content
        assert "00:00:11,000" in content


class TestExportAss:
    def test_header_contains_playres_and_styles(self, tmp_path, sample_segments, config):
        out = tmp_path / "test.ass"
        export_ass(sample_segments, str(out), config=config)
        content = out.read_text()
        assert "PlayResX: 1920" in content
        assert "PlayResY: 1080" in content
        assert "Style: Default,Arial,60" in content
        assert "Style: Caption,Arial,45" in content

    def test_dialogue_uses_default_style(self, tmp_path, sample_segments, config):
        out = tmp_path / "test.ass"
        export_ass(sample_segments, str(out), config=config)
        lines = out.read_text().splitlines()
        dialogue_lines = [l for l in lines if l.startswith("Dialogue:") and ",Default," in l]
        assert len(dialogue_lines) > 0

    def test_captions_use_caption_style(self, tmp_path, sample_segments, config):
        out = tmp_path / "test.ass"
        export_ass(sample_segments, str(out), config=config)
        lines = out.read_text().splitlines()
        caption_lines = [l for l in lines if l.startswith("Dialogue:") and ",Caption," in l]
        assert len(caption_lines) > 0
        # Should have positioning tags
        for line in caption_lines:
            assert "\\pos(" in line

    def test_same_location_captions_stacked(self, tmp_path):
        segments = [
            {
                "start": 0.0,
                "end": 5.0,
                "speaker": "N/A",
                "text": "",
                "english_text": "",
                "on_screen_captions": [
                    {"text": "Line 1", "location": "top-center", "importance": "high"},
                    {"text": "Line 2", "location": "top-center", "importance": "high"},
                ],
            }
        ]
        cfg = TheoriaConfig(line_height=55)
        out = tmp_path / "test.ass"
        export_ass(segments, str(out), config=cfg)
        lines = out.read_text().splitlines()
        caption_lines = [l for l in lines if ",Caption," in l]
        assert len(caption_lines) == 2
        # First caption at y=100, second at y=155 (top alignment stacks downward)
        assert "\\pos(960,100)" in caption_lines[0]
        assert "\\pos(960,155)" in caption_lines[1]
