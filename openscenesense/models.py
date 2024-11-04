from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Optional
import numpy as np


class SceneType(Enum):
    STATIC = "static"
    ACTION = "action"
    TRANSITION = "transition"


@dataclass
class ModelConfig:
    """Configuration for OpenAI models"""
    vision_model: str = "gpt-4-vision-preview"
    text_model: str = "gpt-4-turbo-preview"
    audio_model: str = "whisper-1"


@dataclass
class Frame:
    """Represents a video frame with metadata"""
    image: np.ndarray
    timestamp: float
    scene_type: SceneType
    difference_score: float = 0.0


@dataclass
class AudioSegment:
    """Represents a transcribed segment of audio with timing information"""
    text: str
    start_time: float
    end_time: float
    confidence: float = 0.0


@dataclass
class AnalysisPrompts:
    """Customizable prompts for video and audio analysis"""
    frame_analysis: str = "Describe what's happening in this moment of the video, focusing on important actions, objects, or changes."

    detailed_summary: str = """Create a comprehensive narrative that integrates both visual and audio elements from this {duration:.1f}-second video.

    Video Timeline:
    {timeline}

    Audio Transcript:
    {transcript}

    Provide a detailed summary that combines visual and audio elements into a cohesive story."""

    brief_summary: str = """Based on this {duration:.1f}-second video timeline and transcript:
    {timeline}
    {transcript}

    Provide a concise 2-3 line summary that captures the key events."""
