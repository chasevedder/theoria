"""Configuration system with TOML support and presets."""

from dataclasses import dataclass, field, fields
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]


PRESETS: dict[str, dict[str, str]] = {
    "variety": {
        "intro": "You are an expert Korean to English translator specializing in variety shows.",
        "rules": (
            "IMPORTANT: Korean variety shows often repeat spoken dialogue as on-screen captions. "
            "Do NOT include a caption in 'on_screen_captions' if it simply repeats what the speaker "
            "is currently saying; only include captions that provide unique commentary, context, or "
            "additional information."
        ),
    },
}


@dataclass
class TheoriaConfig:
    # Models
    whisper_model: str = "large-v3"
    gemini_model: str = "gemini-3-flash-preview"

    # Translation
    temperature: float = 1.0
    max_retries: int = 3
    retry_sleep: float = 10.0
    context_window_size: int = 5
    batch_size: int = 30
    max_workers: int = 5

    # Alignment
    max_duration: float = 4.0
    min_gap: float = 0.5

    # Captions
    max_cap_duration: float = 5.0
    line_height: int = 55

    # ASS styles
    play_res_x: int = 1920
    play_res_y: int = 1080
    dialogue_font: str = "Arial"
    dialogue_fontsize: int = 60
    caption_font: str = "Arial"
    caption_fontsize: int = 45

    # Cost (per 1M tokens)
    input_cost_per_million: float = 0.50
    output_cost_per_million: float = 3.00

    # Prompt
    preset: str = "variety"
    custom_prompt: str | None = None

    def get_prompt_intro(self) -> str:
        """Return the system intro line for the current preset or custom prompt."""
        if self.custom_prompt:
            print("Warning: custom_prompt overrides preset intro and genre rules.")
            return self.custom_prompt
        preset_data = PRESETS.get(self.preset, PRESETS["variety"])
        return preset_data["intro"]

    def get_genre_rules(self) -> str:
        """Return genre-specific rules for the current preset."""
        if self.custom_prompt:
            return ""
        preset_data = PRESETS.get(self.preset, PRESETS["variety"])
        return preset_data["rules"]


def load_config(config_path: str | None = None) -> TheoriaConfig:
    """Load config from TOML file(s), with defaults.

    Loading order: defaults -> theoria.toml (cwd then ~/.config/theoria/) -> explicit path -> CLI args (applied later).
    """
    config = TheoriaConfig()
    toml_data: dict = {}

    paths_to_try: list[Path] = []
    if config_path:
        paths_to_try.append(Path(config_path))
    else:
        paths_to_try.append(Path("theoria.toml"))
        paths_to_try.append(Path.home() / ".config" / "theoria" / "theoria.toml")

    for path in paths_to_try:
        if path.exists():
            with open(path, "rb") as f:
                toml_data = tomllib.load(f)
            break

    if not toml_data:
        return config

    # Apply TOML values to the config dataclass
    valid_fields = {f.name for f in fields(TheoriaConfig)}
    for key, value in toml_data.items():
        if key in valid_fields:
            setattr(config, key, value)

    return config
