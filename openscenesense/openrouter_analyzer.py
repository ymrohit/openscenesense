from typing import Dict, Optional
from openai import OpenAI
from .analyzer import VideoAnalyzer
from .models import ModelConfig, AnalysisPrompts
from .frame_selectors import FrameSelector
import logging

class OpenRouterAnalyzer(VideoAnalyzer):
    """Video analyzer that uses OpenRouter for vision/text and OpenAI for audio"""

    def __init__(
            self,
            openrouter_key: str,
            openai_key: str,
            model_config: Optional[ModelConfig] = None,
            frame_selector: Optional[FrameSelector] = None,
            min_frames: int = 8,
            max_frames: int = 32,
            frames_per_minute: float = 4.0,
            prompts: Optional[AnalysisPrompts] = None,
            log_level: int = logging.INFO,
            http_referer: Optional[str] = None,
            app_title: Optional[str] = None
    ):
        """
        Initialize the OpenRouter video analyzer.

        Args:
            openrouter_key: OpenRouter API key for vision and text analysis
            openai_key: OpenAI API key for audio transcription
            model_config: Configuration specifying which models to use
            min_frames: Minimum number of frames to analyze
            max_frames: Maximum number of frames to analyze
            frames_per_minute: Target number of frames to analyze per minute of video
            prompts: Custom prompts for analysis
            log_level: Logging level
            http_referer: Your site URL for OpenRouter rankings
            app_title: Your app name for OpenRouter rankings
        """
        # Initialize with OpenRouter client
        super().__init__(
            api_key=openrouter_key,
            base_url="https://openrouter.ai/api/v1",
            model_config=model_config,
            frame_selector=frame_selector,
            min_frames=min_frames,
            max_frames=max_frames,
            frames_per_minute=frames_per_minute,
            prompts=prompts,
            log_level=log_level
        )

        # Create separate OpenAI client for audio
        self.audio_client = OpenAI(api_key=openai_key)

        # Set OpenRouter headers
        if http_referer or app_title:
            headers = {}
            if http_referer:
                headers["HTTP-Referer"] = http_referer
            if app_title:
                headers["X-Title"] = app_title
            self.client.headers.update(headers)


        self.logger.info(f"Initialized OpenRouterAnalyzer with:"
                         f"\n - Frame selector: {self.frame_selector.__class__.__name__}"
                         f"\n - Vision model: {self.model_config.vision_model}"
                         f"\n - Text model: {self.model_config.text_model}"
                         f"\n - Audio model: {self.model_config.audio_model} (via OpenAI)"
                         )

    def _transcribe_audio(self, video_path: str):
        """Override to use OpenAI client for audio transcription"""
        # Store original client
        original_client = self.client

        try:
            # Temporarily set client to audio_client
            self.client = self.audio_client
            # Call parent's implementation with audio client
            return super()._transcribe_audio(video_path)
        finally:
            # Restore original client
            self.client = original_client