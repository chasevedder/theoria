"""Shared fixtures for theoria tests."""

import pytest

from theoria.config import TheoriaConfig


@pytest.fixture
def config():
    return TheoriaConfig()


@pytest.fixture
def sample_segments():
    """Mixed list of translated segments: dialogue, captions, and combined."""
    return [
        {
            "start": 0.0,
            "end": 3.5,
            "speaker": "Host",
            "text": "안녕하세요",
            "english_text": "Hello everyone",
        },
        {
            "start": 3.5,
            "end": 7.0,
            "speaker": "Guest",
            "text": "반갑습니다",
            "english_text": "Nice to meet you",
            "on_screen_captions": [
                {"text": "First appearance!", "location": "top-center", "importance": "high"},
            ],
        },
        {
            "start": 7.0,
            "end": 10.0,
            "speaker": "N/A",
            "text": "",
            "english_text": "",
            "on_screen_captions": [
                {"text": "Location: Seoul", "location": "bottom-center", "importance": "high"},
                {"text": "wow", "location": "top-center", "importance": "low"},
            ],
        },
    ]
