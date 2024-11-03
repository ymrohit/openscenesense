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
import time

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
        Initialize the video analyzer with configurable models and parameters.

        Args:
            model_config: Configuration specifying which OpenAI models to use
            min_frames: Minimum number of frames to analyze
            max_frames: Maximum number of frames to analyze
            frames_per_minute: Target number of frames to analyze per minute of video
            prompts: Custom prompts for analysis
            log_level: Logging level
        """
        client_kwargs = {}
        if api_key:
            client_kwargs['api_key'] = api_key
        if base_url:
            client_kwargs['base_url'] = base_url
        if organization:
            client_kwargs['organization'] = organization

        self.client = OpenAI(**client_kwargs)
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=log_level)

        self.model_config = model_config or ModelConfig()
        self.frame_selector = frame_selector or DynamicFrameSelector(logger=self.logger)
        self.min_frames = min_frames
        self.max_frames = max_frames
        self.frames_per_minute = frames_per_minute
        self.prompts = prompts or AnalysisPrompts()
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=log_level)

        self.logger.info(f"Initialized VideoAnalyzer with:"
                         f"\n - Frame selector: {self.frame_selector.__class__.__name__}"
                         f"\n - Vision model: {self.model_config.vision_model}"
                         f"\n - Text model: {self.model_config.text_model}"
                         f"\n - Audio model: {self.model_config.audio_model}")

    def _frame_to_base64(self, frame: np.ndarray) -> str:
        """Convert a frame to base64 string"""
        image = Image.fromarray(frame)
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode()

    def _detect_scene_changes(self, cap: cv2.VideoCapture, threshold: float = 20.0) -> List[float]:
        """Detect significant scene changes in the video"""
        scene_changes = []
        prev_frame = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if prev_frame is not None:
                # Calculate frame difference
                gray1 = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
                gray2 = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                diff = cv2.absdiff(gray1, gray2)
                diff_score = np.mean(diff)

                if diff_score > threshold:
                    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    scene_changes.append(timestamp)

            prev_frame = frame

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return scene_changes

    def _select_frames(self, video_path: str) -> List[Frame]:
        """Select key frames from the video"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps

        # Detect scene changes
        scene_changes = self._detect_scene_changes(cap)

        # Calculate optimal number of frames
        base_frames = int(duration / 60 * self.frames_per_minute)
        scene_density = len(scene_changes) / duration if duration > 0 else 0
        scene_multiplier = min(2.0, max(0.5, scene_density * 30))
        target_frames = min(self.max_frames, max(self.min_frames, int(base_frames * scene_multiplier)))

        # Extract frames
        frames = []
        interval = total_frames // target_frames

        for frame_idx in range(0, total_frames, interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Determine scene type
            is_scene_change = any(abs(sc - timestamp) < 0.1 for sc in scene_changes)
            scene_type = SceneType.TRANSITION if is_scene_change else SceneType.STATIC

            frames.append(Frame(
                image=frame_rgb,
                timestamp=timestamp,
                scene_type=scene_type
            ))

        cap.release()
        return frames

    def _transcribe_audio(self, video_path: str) -> List[AudioSegment]:
        """Transcribe audio from video using selected Whisper model"""
        try:
            # Extract audio using librosa
            self.logger.info("Extracting audio from video...")
            audio_array, sr = librosa.load(video_path, sr=16000)

            # Create a temporary file with .wav extension
            import tempfile
            import os

            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                temp_path = temp_audio.name
                self.logger.info(f"Saving temporary audio file to {temp_path}")

                # Save as WAV file using soundfile
                import soundfile as sf
                sf.write(temp_path, audio_array, sr, format='WAV')

            try:
                # Open the temporary file and transcribe
                self.logger.info("Transcribing audio with Whisper API...")
                with open(temp_path, 'rb') as audio_file:
                    response = self.client.audio.transcriptions.create(
                        model=self.model_config.audio_model,
                        file=audio_file,
                        response_format="verbose_json"
                    )

                # Process response
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
                    # Handle case where response doesn't have segments
                    segments.append(AudioSegment(
                        text=response.text if hasattr(response, 'text') else str(response),
                        start_time=0.0,
                        end_time=0.0,
                        confidence=1.0
                    ))

                self.logger.info(f"Successfully transcribed {len(segments)} segments")
                return segments

            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_path)
                    self.logger.debug("Temporary audio file deleted")
                except Exception as e:
                    self.logger.warning(f"Error deleting temporary file: {str(e)}")

        except Exception as e:
            self.logger.error(f"Audio transcription failed: {str(e)}")
            return []

    def _analyze_frame(self, frame: Frame) -> Dict:
        """Analyze a single frame using selected vision model"""
        try:
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

            return {
                "timestamp": frame.timestamp,
                "description": response.choices[0].message.content,
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
        """Generate video summaries using selected text model"""
        # Format timeline and transcript
        timeline = "\n".join(f"Time {desc['timestamp']:.2f}s ({desc['scene_type']}): {desc['description']}"
                             for desc in frame_descriptions)

        transcript = "\n".join(f"[{seg.start_time:.1f}s - {seg.end_time:.1f}s]: {seg.text}"
                               for seg in audio_segments) if audio_segments else "No audio transcript available."

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
        """Perform comprehensive video analysis"""
        self.logger.info(f"Starting video analysis for: {video_path}")

        try:
            # Extract key frames
            #frames = self._select_frames(video_path)
            #self.logger.info(f"Selected {len(frames)} frames for analysis")
            frames = self.frame_selector.select_frames(
                video_path=video_path,
                min_frames=self.min_frames,
                max_frames=self.max_frames,
                frames_per_minute=self.frames_per_minute
            )
            self.logger.info(f"Selected {len(frames)} frames for analysis")
            # Transcribe audio
            audio_segments = self._transcribe_audio(video_path)
            self.logger.info(f"Transcribed {len(audio_segments)} audio segments")

            # Analyze frames
            frame_descriptions = []
            for i, frame in enumerate(frames):
                self.logger.info(f"Analyzing frame {i + 1}/{len(frames)}")
                analysis = self._analyze_frame(frame)
                frame_descriptions.append(analysis)
                time.sleep(0.5)  # Rate limiting

            # Generate summaries
            video_duration = frames[-1].timestamp if frames else 0
            summaries = self._generate_summary(frame_descriptions, audio_segments, video_duration)

            # Prepare result
            result = {
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
                    "scene_distribution": {
                        scene_type.value: len([f for f in frames if f.scene_type == scene_type])
                        for scene_type in SceneType
                    },
                    "models_used": {
                        "vision": self.model_config.vision_model,
                        "text": self.model_config.text_model,
                        "audio": self.model_config.audio_model
                    }
                }
            }

            self.logger.info("Video analysis completed successfully")
            return result

        except Exception as e:
            self.logger.error(f"Error during video analysis: {str(e)}", exc_info=True)
            raise
