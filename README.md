# Prosody-Stretch

**Natural audio duration adjustment for dubbing synchronization.**

A Python library that adjusts TTS audio segment durations while maintaining prosodic naturalness, focused on dubbing synchronization.

## 🎯 The Problem

When TTS models generate audio for dubbing, the duration often differs from the original audio due to:
- Length differences between languages after translation
- Variable speed of synthesized voice
- Distinct prosodic characteristics

Simple uniform speed-up/slow-down techniques produce artificial results.

## ✨ The Solution

Prosody-Stretch applies duration adjustments **non-uniformly**, distributing modifications across:
- **Pauses** between words (most natural)
- **Time-stretch** (WSOLA) on speech segments
- **Vowel extension** at word endings

## 📦 Installation

```bash
# Basic installation
pip install prosody-stretch

# With API support
pip install prosody-stretch[api]

# Full installation
pip install prosody-stretch[full]
```

### From Source

```bash
git clone https://github.com/wallaceblaia/prosody-stretch.git
cd prosody-stretch
pip install -e .
```

## 🔗 Dependencies

### Core Dependencies
- **[NumPy](https://numpy.org/)** - Array processing
- **[SciPy](https://scipy.org/)** - Signal processing
- **[librosa](https://librosa.org/)** - Audio analysis
- **[SoundFile](https://pysoundfile.readthedocs.io/)** - Audio I/O
- **[pytsmod](https://github.com/KAIST-MACLab/pytsmod)** - Time-scale modification (WSOLA)
- **[Click](https://click.palletsprojects.com/)** - CLI framework

### Optional Dependencies
- **[FastAPI](https://fastapi.tiangolo.com/)** - REST API (install with `[api]`)
- **[uvicorn](https://www.uvicorn.org/)** - ASGI server (install with `[api]`)
- **[PyTorch](https://pytorch.org/)** - Deep learning (install with `[full]`)

## 🚀 Quick Start

### Python API

```python
from prosody_stretch import ProsodyStretcher

stretcher = ProsodyStretcher()

# Adjust to specific duration (in seconds)
audio, report = stretcher.adjust_duration(
    audio_path="input.wav",
    target_duration=15.87
)

print(f"Final duration: {report.final_duration:.3f}s")
print(f"Quality: {report.quality_score:.2f}")

# Match reference audio duration
audio, report = stretcher.match_duration(
    source_audio="dubbed.wav",
    reference_audio="original.wav"
)
```

### Command Line Interface (CLI)

```bash
# Adjust to specific duration
prosody-stretch adjust input.wav --target 15.87 -o output.wav

# Match reference duration
prosody-stretch match dubbed.wav original.wav -o output.wav

# Show audio information
prosody-stretch info input.wav

# Start API server
prosody-stretch serve --port 8000
```

### REST API

```bash
# Start server
prosody-stretch serve

# Adjust duration via API
curl -X POST "http://localhost:8000/adjust" \
  -F "file=@input.wav" \
  -F "target_duration=15.87" \
  -o output.wav

# Get audio info
curl -X POST "http://localhost:8000/info" \
  -F "file=@input.wav"
```

## 📊 Adjustment Limits

| Adjustment | Recommended | Maximum | Quality |
|------------|-------------|---------|---------|
| ±10% | ✅ Excellent | - | 0.95+ |
| ±20% | ✅ Good | - | 0.85+ |
| ±30% | ⚠️ Acceptable | - | 0.70+ |
| ±40% | ⚠️ Limit | Maximum | 0.60+ |

## 🔧 Configuration

```python
from prosody_stretch import ProsodyStretcher

stretcher = ProsodyStretcher(
    quality_threshold=0.6,  # Minimum acceptable quality
    prefer_pauses=True,     # Prioritize pause manipulation
    sample_rate=22050       # Target sample rate
)
```

## 📚 API Reference

### `ProsodyStretcher`

#### `adjust_duration(audio_path, target_duration, text=None, output_path=None)`
Adjusts audio duration to target value.

**Parameters:**
- `audio_path`: Path to audio file
- `target_duration`: Target duration in seconds
- `text`: Optional transcription (improves quality)
- `output_path`: Optional output path

**Returns:** `(audio_array, AdjustmentReport)`

#### `match_duration(source_audio, reference_audio, text=None)`
Adjusts source audio to match reference audio duration.

#### `adjust_duration_array(audio, sr, target_duration, text=None)`
Version that accepts numpy array directly.

### `AdjustmentReport`

```python
report.original_duration  # Original duration
report.target_duration    # Target duration
report.final_duration     # Final duration
report.quality_score      # Quality score (0-1)
report.strategies_used    # List of strategies used
report.warnings           # List of warnings
```

## 🛠️ Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Formatting
black prosody_stretch/
ruff check prosody_stretch/
```

## 📄 License

MIT License
