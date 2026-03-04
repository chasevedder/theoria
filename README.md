# theoria
> [!NOTE]
> This project was built almost entirely with AI-assisted code generation using [Claude Code](https://claude.com/claude-code).

Multimodal AI subtitle generator. Takes a foreign-language video and produces translated subtitles by combining speech recognition (Whisper), speaker diarization (Pyannote), video frame analysis, and LLM translation (Gemini).

The pipeline:
1. Extracts audio and runs Whisper for word-level transcription
2. Runs Pyannote speaker diarization to identify who's talking
3. Aligns words to speakers and extracts video frames at segment midpoints
4. Sends frames + transcript to Gemini for translation, speaker identification, and on-screen caption extraction
5. Exports to SRT and/or ASS with positioned captions

## Requirements

- Python 3.12
- CUDA-capable GPU (recommended)
- FFmpeg
- API keys:
  - `GEMINI_API_KEY` — [Google AI Studio](https://aistudio.google.com/)
  - `HF_TOKEN` — [Hugging Face](https://huggingface.co/settings/tokens) (for Pyannote model access)

## Installation

```bash
git clone https://github.com/yourusername/theoria.git
cd theoria
uv sync
source .venv/bin/activate
```

## Usage

```bash
# Basic — Korean video to SRT
theoria -v episode.mkv

# Multiple output formats
theoria -v episode.mkv --format srt ass

# Sequential mode for better name consistency across the episode
theoria -v episode.mkv --sequential

# Limit to first 50 segments for a quick test
theoria -v episode.mkv --limit-segments 50 --no-cleanup
```

### CLI Options

| Flag | Description | Default |
|---|---|---|
| `-v, --video` | Path to source video (required) | — |
| `-o, --output` | Output file path (single format only) | auto-derived |
| `--format` | Output format(s): `srt`, `ass`, or both | `srt` |
| `--lang` | Source language code | `ko` |
| `--preset` | Translation preset (currently: `variety`) | `variety` |
| `--sequential` | Process batches sequentially for context continuity | off |
| `--batch-size` | Segments per Gemini API call | `30` |
| `--max-workers` | Parallel API workers | `5` |
| `--gemini-model` | Gemini model ID | `gemini-3-flash-preview` |
| `--sample-rate` | Fixed frame sampling interval (seconds, 0 = dialogue midpoints) | `0` |
| `--detect-scenes` | Use scene detection for frame selection | off |
| `--sample-duration` | Limit audio extraction to N seconds (testing) | — |
| `--limit-segments` | Process only the first N segments (testing) | — |
| `--no-cleanup` | Keep temporary files after completion | off |
| `--config` | Path to a `theoria.toml` config file | auto-discovered |

## Configuration

All CLI defaults can be overridden via a `theoria.toml` file. Loading order:

1. Built-in defaults
2. `theoria.toml` in the current directory
3. `~/.config/theoria/theoria.toml`
4. Explicit `--config` path
5. CLI arguments (highest priority)

Copy the example to get started:

```bash
cp theoria.toml.example theoria.toml
```

See [`theoria.toml.example`](theoria.toml.example) for all available options.

### Presets

Presets control the translation prompt intro and genre-specific rules. Currently only **variety** is built-in (Korean variety shows).

For other genres or languages, set `custom_prompt` in `theoria.toml` to fully replace the intro and genre rules while keeping the core structural rules (speaker ID, JSON schema, etc.) intact.

## Output Formats

**SRT** — Standard subtitle format. Dialogue and high-importance on-screen captions are written as separate timed entries.

**ASS** — Advanced SubStation Alpha. Dialogue uses a Default style at the bottom. On-screen captions are positioned at their detected screen location using the Caption style, with automatic stacking to prevent overlap. **Note:** ASS output is experimental — caption positioning relies on Gemini's location estimates, which can be inconsistent, which can lead to captions overlapping or not aligning well with their on-screen source.

## Project Structure

```
theoria/
  __init__.py      # Version, warning suppression
  __main__.py      # python -m theoria support
  cli.py           # Argument parsing, config loading, entry point
  config.py        # TheoriaConfig dataclass, TOML loading, presets
  types.py         # Segment/Caption TypedDicts
  audio.py         # FFmpeg audio extraction
  alignment.py     # Whisper + Pyannote timestamp alignment
  translation.py   # Gemini translation + response validation
  exporters.py     # SRT/ASS file writers
  pipeline.py      # Main processing pipeline
```
