"""Shared type definitions."""

from typing import TypedDict


class Segment(TypedDict, total=False):
    start: float
    end: float
    speaker: str
    text: str
    english_text: str
    on_screen_captions: list[dict]


class Caption(TypedDict, total=False):
    text: str
    location: str
    importance: str
