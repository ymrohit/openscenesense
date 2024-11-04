from typing import Dict, Optional, List
import logging
from openai import OpenAI
from .models import ModelConfig, AnalysisPrompts, AudioSegment, Frame, SceneType
from .frame_selectors import FrameSelector, DynamicFrameSelector
import cv2
import numpy as np
from PIL import Image
import io
import base64
import librosa
import tempfile
import os
import soundfile as sf
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


class VideoAnalyzerError(Exception):
    """Base exception class for VideoAnalyzer."""
    pass


class VideoAnalysisError(VideoAnalyzerError):
    """Exception raised for errors during video analysis."""
    pass


class AudioTranscriptionError(VideoAnalyzerError):
    """Exception raised for errors during audio transcription."""
    pass


class FrameAnalysisError(VideoAnalyzerError):
    """Exception raised for errors during frame analysis."""
    pass


class VideoAnalyzer:
    def __init__(
            self,
            api_key: Optional[str] = None,
            model_config: Optional[ModelConfig] = None,
            frame_selector: Optional[FrameSelector] = None,
            min_frames: int = 8,
            max_frames: int = 32,
            frames_per_minute: float = 4.0,
            prompts: Optional[AnalysisPrompts] = None,
            log_level: int = logging.INFO,
            base_url: Optional[str] = None,
            organization: Optional[str] = None
    ):
        """
        Initialize the VideoAnalyzer with configurable models and parameters.

        Args:
            api_key: OpenAI API key.
            model_config: Configuration specifying which OpenAI models to use.
            frame_selector: Strategy for selecting frames from the video.
            min_frames: Minimum number of frames to analyze.
            max_frames: Maximum number of frames to analyze.
            frames_per_minute: Target number of frames to analyze per minute of video.
            prompts: Custom prompts for analysis.
            log_level: Logging level.
            base_url: Base URL for the OpenAI API.
            organization: OpenAI organization ID.
        """
        self._configure_logging(log_level)
        self.logger = logging.getLogger(__name__)

        self._validate_initialization_parameters(
            api_key, model_config, frame_selector,
            min_frames, max_frames, frames_per_minute
        )

        self.client = self._initialize_openai_client(api_key, base_url, organization)

        self.model_config = model_config or ModelConfig()
        self.frame_selector = frame_selector or DynamicFrameSelector(logger=self.logger)
        self.min_frames = min_frames
        self.max_frames = max_frames
        self.frames_per_minute = frames_per_minute
        self.prompts = prompts or AnalysisPrompts()

        self.logger.info(f"Initialized VideoAnalyzer with:"
                         f"\n - Frame selector: {self.frame_selector.__class__.__name__}"
                         f"\n - Vision model: {self.model_config.vision_model}"
                         f"\n - Text model: {self.model_config.text_model}"
                         f"\n - Audio model: {self.model_config.audio_model}")

    def _configure_logging(self, log_level: int) -> None:
        """Configure logging settings."""
        logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')

    def _validate_initialization_parameters(
            self,
            api_key: Optional[str],
            model_config: Optional[ModelConfig],
            frame_selector: Optional[FrameSelector],
            min_frames: int,
            max_frames: int,
            frames_per_minute: float
    ) -> None:
        """Validate initialization parameters."""
        if min_frames < 1:
            raise ValueError("min_frames must be at least 1.")
        if max_frames < min_frames:
            raise ValueError("max_frames must be greater than or equal to min_frames.")
        if frames_per_minute <= 0:
            raise ValueError("frames_per_minute must be a positive number.")

    def _initialize_openai_client(self, api_key: Optional[str], base_url: Optional[str],
                                  organization: Optional[str]) -> OpenAI:
        """Initialize the OpenAI client with provided credentials."""
        client_kwargs = {}
        if api_key:
            client_kwargs['api_key'] = api_key
        if base_url:
            client_kwargs['base_url'] = base_url
        if organization:
            client_kwargs['organization'] = organization

        try:
            client = OpenAI(**client_kwargs)
            self.logger.debug("OpenAI client initialized successfully.")
            return client
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI client: {str(e)}")
            raise VideoAnalyzerError("OpenAI client initialization failed.") from e

    def _frame_to_base64(self, frame: np.ndarray) -> str:
        """Convert a frame to a base64-encoded JPEG string."""
        try:
            image = Image.fromarray(frame)
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            base64_image = base64.b64encode(buffered.getvalue()).decode()
            self.logger.debug("Frame converted to base64 successfully.")
            return base64_image
        except Exception as e:
            self.logger.error(f"Failed to convert frame to base64: {str(e)}")
            raise FrameAnalysisError("Frame conversion to base64 failed.") from e

    def _detect_scene_changes(self, cap: cv2.VideoCapture, threshold: float = 20.0) -> List[float]:
        """
        Detect significant scene changes in the video.

        Args:
            cap: OpenCV VideoCapture object.
            threshold: Difference threshold to consider a scene change.

        Returns:
            List of timestamps (in seconds) where scene changes occur.
        """
        scene_changes = []
        prev_frame = None

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                if prev_frame is not None:
                    # Calculate frame difference
                    gray1 = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
                    gray2 = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
                    diff = cv2.absdiff(gray1, gray2)
                    diff_score = np.mean(diff)

                    if diff_score > threshold:
                        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                        scene_changes.append(timestamp)
                        self.logger.debug(f"Scene change detected at {timestamp:.2f}s with diff score {diff_score:.2f}")

                prev_frame = frame_rgb

            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.logger.info(f"Detected {len(scene_changes)} scene changes.")
            return scene_changes

        except Exception as e:
            self.logger.error(f"Scene change detection failed: {str(e)}")
            raise VideoAnalysisError("Scene change detection failed.") from e

    def _transcribe_audio(self, video_path: str) -> List[AudioSegment]:
        """
        Transcribe audio from video using the selected Whisper model.

        Args:
            video_path: Path to the video file.

        Returns:
            List of transcribed audio segments.
        """
        try:
            self.logger.info("Extracting audio from video...")
            audio_array, sr = librosa.load(video_path, sr=16000)

            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                temp_path = temp_audio.name
                self.logger.debug(f"Saving temporary audio file to {temp_path}")
                sf.write(temp_path, audio_array, sr, format='WAV')

            try:
                self.logger.info("Transcribing audio with Whisper API...")
                with open(temp_path, 'rb') as audio_file:
                    response = self.client.audio.transcriptions.create(
                        model=self.model_config.audio_model,
                        file=audio_file,
                        response_format="verbose_json"
                    )

                segments = []
                if hasattr(response, 'segments'):
                    for segment in response.segments:
                        segments.append(AudioSegment(
                            text=segment.text,
                            start_time=segment.start,
                            end_time=segment.end,
                            confidence=segment.confidence if hasattr(segment, 'confidence') else 1.0
                        ))
                else:
                    segments.append(AudioSegment(
                        text=response.text if hasattr(response, 'text') else str(response),
                        start_time=0.0,
                        end_time=0.0,
                        confidence=1.0
                    ))

                self.logger.info(f"Successfully transcribed {len(segments)} audio segments.")
                return segments

            except Exception as e:
                self.logger.error(f"Audio transcription failed: {str(e)}")
                #raise AudioTranscriptionError("Audio transcription failed.") from e
                return []

            finally:
                try:
                    os.unlink(temp_path)
                    self.logger.debug("Temporary audio file deleted successfully.")
                except Exception as e:
                    self.logger.warning(f"Failed to delete temporary audio file: {str(e)}")

        except Exception as e:
            self.logger.error(f"Audio extraction failed: {str(e)}")
            #raise AudioTranscriptionError("Audio extraction failed.") from e
            return []

    def _analyze_frame(self, frame: Frame) -> Dict:
        """
        Analyze a single frame using the selected vision model.

        Args:
            frame: Frame object containing image data and metadata.

        Returns:
            Dictionary containing analysis results.
        """
        try:
            if frame.image is None:
                raise ValueError("Frame image data is missing.")

            base64_image = self._frame_to_base64(frame.image)

            response = self.client.chat.completions.create(
                model=self.model_config.vision_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.prompts.frame_analysis},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=300
            )

            if not response or not response.choices:
                raise FrameAnalysisError("Received empty response or choices from API.")

            message = response.choices[0].message
            if not hasattr(message, 'content'):
                raise FrameAnalysisError("Response message does not contain 'content' attribute.")

            description = message.content
            if not description:
                raise FrameAnalysisError("No content found in response message.")

            self.logger.debug(f"Frame at {frame.timestamp:.2f}s analyzed successfully.")
            return {
                "timestamp": frame.timestamp,
                "description": description,
                "scene_type": frame.scene_type.value
            }

        except Exception as e:
            self.logger.error(f"Frame analysis failed: {str(e)}")
            return {
                "timestamp": frame.timestamp,
                "description": "Error analyzing frame",
                "scene_type": frame.scene_type.value
            }

    def _generate_summary(self, frame_descriptions: List[Dict], audio_segments: List[AudioSegment],
                          video_duration: float) -> Dict:
        """
        Generate video summaries using the selected text model.

        Args:
            frame_descriptions: List of frame analysis results.
            audio_segments: List of transcribed audio segments.
            video_duration: Total duration of the video in seconds.

        Returns:
            Dictionary containing detailed and brief summaries, timeline, and transcript.
        """
        timeline = "\n".join(
            f"Time {desc['timestamp']:.2f}s ({desc['scene_type']}): {desc['description']}"
            for desc in frame_descriptions
        )

        transcript = "\n".join(
            f"[{seg.start_time:.1f}s - {seg.end_time:.1f}s]: {seg.text}"
            for seg in audio_segments
        ) if audio_segments else "No audio transcript available."

        try:
            # Generate detailed summary
            detailed_response = self.client.chat.completions.create(
                model=self.model_config.text_model,
                messages=[{
                    "role": "user",
                    "content": self.prompts.detailed_summary.format(
                        duration=video_duration,
                        timeline=timeline,
                        transcript=transcript
                    )
                }]
            )

            # Generate brief summary
            brief_response = self.client.chat.completions.create(
                model=self.model_config.text_model,
                messages=[{
                    "role": "user",
                    "content": self.prompts.brief_summary.format(
                        duration=video_duration,
                        timeline=timeline,
                        transcript=transcript
                    )
                }]
            )

            self.logger.debug("Summaries generated successfully.")
            return {
                "detailed": detailed_response.choices[0].message.content,
                "brief": brief_response.choices[0].message.content,
                "timeline": timeline,
                "transcript": transcript
            }

        except Exception as e:
            self.logger.error(f"Summary generation failed: {str(e)}")
            return {
                "detailed": "Error generating detailed summary",
                "brief": "Error generating brief summary",
                "timeline": timeline,
                "transcript": transcript
            }
    def analyze_video(self, video_path: str) -> Dict:
        """
        Perform comprehensive video analysis.

        Args:
            video_path: Path to the video file.

        Returns:
            Dictionary containing summaries, frame analyses, audio segments, and metadata.
        """
        self.logger.info(f"Starting video analysis for: {video_path}")

        if not os.path.isfile(video_path):
            self.logger.error(f"Video file does not exist: {video_path}")
            raise VideoAnalysisError(f"Video file does not exist: {video_path}")

        try:
            frames = self._select_and_log_frames(video_path)
            audio_segments = self._transcribe_audio(video_path)
            frame_descriptions = self._concurrently_analyze_frames(frames)
            video_duration = frames[-1].timestamp if frames else 0.0
            summaries = self._generate_summary(frame_descriptions, audio_segments, video_duration)
            result = self._compile_results(frames, frame_descriptions, audio_segments, summaries, video_duration)

            self.logger.info("Video analysis completed successfully.")
            return result

        except VideoAnalyzerError as e:
            self.logger.error(f"Video analysis failed: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error during video analysis: {str(e)}", exc_info=True)
            raise VideoAnalysisError("Unexpected error during video analysis.") from e

    def _select_and_log_frames(self, video_path: str) -> List[Frame]:
        """Select key frames using the frame selector and log the count."""
        frames = self.frame_selector.select_frames(
            video_path=video_path,
            min_frames=self.min_frames,
            max_frames=self.max_frames,
            frames_per_minute=self.frames_per_minute
        )
        self.logger.info(f"Selected {len(frames)} frames for analysis.")
        return frames

    def _concurrently_analyze_frames(self, frames: List[Frame]) -> List[Dict]:
        """Analyze frames concurrently to improve performance."""
        frame_descriptions = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_frame = {
                executor.submit(self._analyze_frame, frame): frame for frame in frames
            }
            for future in as_completed(future_to_frame):
                frame = future_to_frame[future]
                try:
                    analysis = future.result()
                    frame_descriptions.append(analysis)
                    self.logger.debug(f"Analyzed frame at {analysis['timestamp']:.2f}s.")
                except Exception as e:
                    self.logger.error(f"Error analyzing frame at {frame.timestamp}s: {str(e)}")
                    frame_descriptions.append({
                        "timestamp": frame.timestamp,
                        "description": "Error analyzing frame",
                        "scene_type": frame.scene_type.value
                    })
                time.sleep(0.1)  # Minimal delay to respect rate limits

        # Sort frame_descriptions by timestamp to maintain order
        frame_descriptions_sorted = sorted(frame_descriptions, key=lambda x: x['timestamp'])
        self.logger.debug("Frame descriptions sorted by timestamp.")
        return frame_descriptions_sorted

    def _compile_results(self, frames: List[Frame], frame_descriptions: List[Dict],
                        audio_segments: List[AudioSegment], summaries: Dict, video_duration: float) -> Dict:
        """Compile all analysis results into a single dictionary."""
        scene_distribution = {
            scene_type.value: len([f for f in frames if f.scene_type == scene_type])
            for scene_type in SceneType
        }

        return {
            "summary": summaries["detailed"],
            "brief_summary": summaries["brief"],
            "timeline": summaries["timeline"],
            "transcript": summaries["transcript"],
            "frame_analyses": frame_descriptions,
            "audio_segments": [
                {
                    "text": segment.text,
                    "start_time": segment.start_time,
                    "end_time": segment.end_time,
                    "confidence": segment.confidence
                }
                for segment in audio_segments
            ],
            "metadata": {
                "num_frames_analyzed": len(frames),
                "num_audio_segments": len(audio_segments),
                "video_duration": video_duration,
                "scene_distribution": scene_distribution,
                "models_used": {
                    "vision": self.model_config.vision_model,
                    "text": self.model_config.text_model,
                    "audio": self.model_config.audio_model
                }
            }
        }
