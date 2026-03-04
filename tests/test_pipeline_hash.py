"""Tests for _hash() helper and hash chain cascade properties."""

from theoria.hashing import _hash


class TestHash:
    def test_deterministic(self):
        assert _hash("a", "b", "c") == _hash("a", "b", "c")

    def test_returns_64_char_hex(self):
        result = _hash("x")
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_different_inputs_differ(self):
        assert _hash("a") != _hash("b")

    def test_order_matters(self):
        assert _hash("x", "y") != _hash("y", "x")

    def test_empty_extra_arg_differs(self):
        assert _hash("a") != _hash("a", "")

    def test_numeric_and_string_equivalent(self):
        # _hash converts all parts to str, so 1 and "1" must match
        assert _hash(1, 2) == _hash("1", "2")


class TestHashChain:
    """Verify that the hash chain propagates changes correctly.

    The chain mirrors the computation in run_pipeline:
        diar    <- audio_size
        whisper <- diar, lang, whisper_model
        align   <- diar, whisper, max_duration, min_gap
        frame   <- align, video_mtime, video_size, sample_rate, detect_scenes
        trans   <- frame, gemini_model, temperature, preset, custom_prompt,
                   batch_size, sequential, context_window_size, limit_segments
    """

    def _chain(
        self,
        audio_size=1000,
        lang="ko",
        whisper_model="large-v3",
        max_duration=4.0,
        min_gap=0.5,
        video_mtime=1000.0,
        video_size=50000,
        sample_rate=0,
        detect_scenes=False,
        gemini_model="gemini-2.0-flash",
        temperature=1.0,
        preset="variety",
        custom_prompt="",
        batch_size=10,
        sequential=False,
        context_window_size=5,
        limit_segments=0,
    ):
        diar = _hash(audio_size)
        whisper = _hash(diar, lang, whisper_model)
        align = _hash(diar, whisper, max_duration, min_gap)
        frame = _hash(align, video_mtime, video_size, sample_rate, detect_scenes)
        trans = _hash(
            frame, gemini_model, temperature, preset, custom_prompt,
            batch_size, sequential, context_window_size, limit_segments,
        )
        return diar, whisper, align, frame, trans

    def test_changing_lang_leaves_diar_unchanged(self):
        d1, w1, a1, f1, t1 = self._chain(lang="ko")
        d2, w2, a2, f2, t2 = self._chain(lang="ja")
        assert d1 == d2
        assert w1 != w2
        assert a1 != a2
        assert f1 != f2
        assert t1 != t2

    def test_changing_audio_size_cascades_all(self):
        d1, w1, a1, f1, t1 = self._chain(audio_size=1000)
        d2, w2, a2, f2, t2 = self._chain(audio_size=9999)
        assert d1 != d2
        assert w1 != w2
        assert a1 != a2
        assert f1 != f2
        assert t1 != t2

    def test_changing_gemini_model_only_changes_trans(self):
        d1, w1, a1, f1, t1 = self._chain(gemini_model="gemini-2.0-flash")
        d2, w2, a2, f2, t2 = self._chain(gemini_model="gemini-1.5-pro")
        assert d1 == d2
        assert w1 == w2
        assert a1 == a2
        assert f1 == f2
        assert t1 != t2

    def test_changing_sample_rate_leaves_diar_whisper_align_unchanged(self):
        d1, w1, a1, f1, t1 = self._chain(sample_rate=0)
        d2, w2, a2, f2, t2 = self._chain(sample_rate=2.0)
        assert d1 == d2
        assert w1 == w2
        assert a1 == a2
        assert f1 != f2
        assert t1 != t2

    def test_changing_detect_scenes_leaves_diar_whisper_align_unchanged(self):
        d1, w1, a1, f1, t1 = self._chain(detect_scenes=False)
        d2, w2, a2, f2, t2 = self._chain(detect_scenes=True)
        assert d1 == d2
        assert w1 == w2
        assert a1 == a2
        assert f1 != f2
        assert t1 != t2

    def test_changing_video_stats_leaves_diar_whisper_align_unchanged(self):
        d1, w1, a1, f1, t1 = self._chain(video_mtime=100.0, video_size=1000)
        d2, w2, a2, f2, t2 = self._chain(video_mtime=200.0, video_size=2000)
        assert d1 == d2
        assert w1 == w2
        assert a1 == a2
        assert f1 != f2
        assert t1 != t2

    def test_changing_whisper_model_leaves_diar_unchanged(self):
        d1, w1, a1, f1, t1 = self._chain(whisper_model="large-v3")
        d2, w2, a2, f2, t2 = self._chain(whisper_model="medium")
        assert d1 == d2
        assert w1 != w2
        assert a1 != a2
        assert f1 != f2
        assert t1 != t2

    def test_changing_max_duration_leaves_diar_whisper_unchanged(self):
        d1, w1, a1, f1, t1 = self._chain(max_duration=4.0)
        d2, w2, a2, f2, t2 = self._chain(max_duration=8.0)
        assert d1 == d2
        assert w1 == w2
        assert a1 != a2
        assert f1 != f2
        assert t1 != t2

    def test_changing_limit_segments_only_changes_trans(self):
        d1, w1, a1, f1, t1 = self._chain(limit_segments=0)
        d2, w2, a2, f2, t2 = self._chain(limit_segments=50)
        assert d1 == d2
        assert w1 == w2
        assert a1 == a2
        assert f1 == f2
        assert t1 != t2

    def test_changing_sequential_only_changes_trans(self):
        d1, w1, a1, f1, t1 = self._chain(sequential=False)
        d2, w2, a2, f2, t2 = self._chain(sequential=True)
        assert d1 == d2
        assert w1 == w2
        assert a1 == a2
        assert f1 == f2
        assert t1 != t2
