from openscenesense.models import ModelConfig, Frame, AudioSegment, AnalysisPrompts, SceneType
from openscenesense.analyzer import VideoAnalyzer
from openscenesense.openrouter_analyzer import OpenRouterAnalyzer
from openscenesense.frame_selectors import FrameSelector, DynamicFrameSelector, UniformFrameSelector

import subprocess
import sys

def check_ffmpeg():
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        print("FFmpeg is not installed. Please install FFmpeg before using this package.", file=sys.stderr)
        print("Installation instructions can be found in the package README.", file=sys.stderr)
        return False

# Light check at import time; warn but don't raise to keep package importable
try:
    if not check_ffmpeg():
        print("Warning: FFmpeg not found. Audio transcription may be unavailable.", file=sys.stderr)
except Exception:
    # Never fail import due to environment checks
    pass

__version__ = "1.1.0"

__all__ = [
    'VideoAnalyzer',
    'OpenRouterAnalyzer',
    'ModelConfig',
    'Frame',
    'AudioSegment',
    'AnalysisPrompts',
    'SceneType',
    'FrameSelector',
    'DynamicFrameSelector',
    'UniformFrameSelector',
]
