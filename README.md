# OpenSceneSense

A Python package for analyzing videos using OpenAI and Openrouter Vision models.

## Prerequisites

This package requires FFmpeg to be installed on your system:

### Ubuntu/Debian
```bash
sudo apt update
sudo apt install ffmpeg
```

### macOS (using Homebrew)
```bash
brew install ffmpeg
```

### Windows
1. Download FFmpeg from https://ffmpeg.org/download.html
2. Extract the archive
3. Add the bin folder to your system PATH

You can verify the installation by running:
```bash
ffmpeg -version
```

## Installation

```bash
pip install openscenesense
```

## Usage

```python
from openscenesense import VideoAnalyzer

analyzer = VideoAnalyzer(api_key="your-openai-api-key")
result = analyzer.analyze_video("path/to/video.mp4")
```

## Features

- Frame analysis using GPT Vision Models
- Audio transcription using Whisper
- Dynamic frame selection
- Scene change detection
- Comprehensive video summaries

## Requirements

- Python 3.10+
- FFmpeg
- OpenAI/Openrouter API key
- Dependencies listed in pyproject.toml
```