"""Tests for TheoriaConfig and load_config."""

from theoria.config import TheoriaConfig, load_config, PRESETS


class TestTheoriaConfigDefaults:
    def test_default_whisper_model(self, config):
        assert config.whisper_model == "large-v3"

    def test_default_gemini_model(self, config):
        assert config.gemini_model == "gemini-3-flash-preview"

    def test_default_numeric_ranges(self, config):
        assert config.temperature == 1.0
        assert config.max_retries == 3
        assert config.batch_size == 30
        assert config.max_duration == 4.0
        assert config.min_gap == 0.5
        assert config.max_cap_duration == 5.0
        assert config.line_height == 55

    def test_default_costs_from_config_false(self, config):
        assert config._costs_from_config is False


class TestGetPromptIntro:
    def test_returns_variety_intro_by_default(self, config):
        intro = config.get_prompt_intro()
        assert intro == PRESETS["variety"]["intro"]

    def test_returns_custom_prompt_when_set(self):
        config = TheoriaConfig(custom_prompt="You are a drama translator.")
        assert config.get_prompt_intro() == "You are a drama translator."


class TestGetGenreRules:
    def test_returns_variety_rules_by_default(self, config):
        rules = config.get_genre_rules()
        assert rules == PRESETS["variety"]["rules"]

    def test_returns_empty_when_custom_prompt_set(self):
        config = TheoriaConfig(custom_prompt="Custom prompt")
        assert config.get_genre_rules() == ""


class TestLoadConfig:
    def test_returns_defaults_when_no_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cfg = load_config()
        assert cfg.whisper_model == "large-v3"
        assert cfg._costs_from_config is False

    def test_loads_toml_values(self, tmp_path):
        toml_file = tmp_path / "test.toml"
        toml_file.write_text('whisper_model = "small"\nbatch_size = 10\n')
        cfg = load_config(str(toml_file))
        assert cfg.whisper_model == "small"
        assert cfg.batch_size == 10

    def test_sets_costs_from_config_flag(self, tmp_path):
        toml_file = tmp_path / "test.toml"
        toml_file.write_text("input_cost_per_million = 1.0\n")
        cfg = load_config(str(toml_file))
        assert cfg._costs_from_config is True

    def test_toml_overrides_defaults(self, tmp_path):
        toml_file = tmp_path / "test.toml"
        toml_file.write_text('preset = "variety"\ntemperature = 0.5\nmax_duration = 6.0\n')
        cfg = load_config(str(toml_file))
        assert cfg.temperature == 0.5
        assert cfg.max_duration == 6.0
        # Non-overridden values stay default
        assert cfg.max_retries == 3

    def test_unknown_keys_ignored(self, tmp_path):
        toml_file = tmp_path / "test.toml"
        toml_file.write_text('fake_key = "nope"\nwhisper_model = "tiny"\n')
        cfg = load_config(str(toml_file))
        assert cfg.whisper_model == "tiny"
        assert not hasattr(cfg, "fake_key") or getattr(cfg, "fake_key", None) is None
