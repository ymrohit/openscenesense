from src.openscenesense.models import ModelConfig, Frame, AudioSegment, AnalysisPrompts, SceneType
from src.openscenesense.analyzer import VideoAnalyzer
from src.openscenesense.openrouter_analyzer import OpenRouterAnalyzer
from src.openscenesense.frame_selectors import FrameSelector, DynamicFrameSelector, UniformFrameSelector

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

# Add this check at package initialization or before video processing
if not check_ffmpeg():
    raise RuntimeError("FFmpeg is required but not found on the system.")

__version__ = "1.0.0"

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