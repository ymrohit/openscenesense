from abc import ABC, abstractmethod
import cv2
import numpy as np
from typing import List
import logging
from .models import Frame,SceneType


class FrameSelector(ABC):
    """Abstract base class for frame selection strategies"""

    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def select_frames(self, video_path: str, min_frames: int, max_frames: int,
                      frames_per_minute: float) -> List[Frame]:
        """Select frames from video according to strategy"""
        pass

class DynamicFrameSelector(FrameSelector):
    """Selects frames based on scene changes and motion detection"""

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

    def select_frames(self, video_path: str, min_frames: int, max_frames: int,
                      frames_per_minute: float) -> List[Frame]:
        """Select key frames based on scene changes and motion"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video file: {video_path}")

        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps

            # Detect scene changes
            self.logger.info("Detecting scene changes...")
            scene_changes = self._detect_scene_changes(cap)

            # Debug logging
            self.logger.info(f"Video duration: {duration:.2f} seconds")
            self.logger.info(f"Detected {len(scene_changes)} scene changes")
            self.logger.info(f"Frames per minute setting: {frames_per_minute}")

            # Calculate target frames based on duration and scene changes
            base_frames = max(min_frames, int(duration / 60 * frames_per_minute))

            # Adjust frame count based on scene density
            scene_density = len(scene_changes) / duration if duration > 0 else 0
            scene_multiplier = min(2.0, max(1.0, scene_density))  # Modified multiplier logic
            target_frames = min(max_frames, int(base_frames * scene_multiplier))

            self.logger.info(f"Base frames: {base_frames}")
            self.logger.info(f"Scene density: {scene_density:.2f}")
            self.logger.info(f"Scene multiplier: {scene_multiplier:.2f}")
            self.logger.info(f"Target frames: {target_frames}")

            # If video is very short, ensure minimum frames
            if duration < 60 and target_frames < min_frames:
                target_frames = min_frames
                self.logger.info(f"Short video, adjusting to minimum frames: {target_frames}")

            # Calculate frame interval based on target frames
            interval = max(1, total_frames // target_frames)

            # Extract frames
            frames = []
            prev_timestamp = -1

            for frame_idx in range(0, total_frames, interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    break

                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

                # Skip if too close to previous frame
                if prev_timestamp >= 0 and (timestamp - prev_timestamp) < 0.5:
                    continue

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Determine scene type and calculate difference score
                is_scene_change = any(abs(sc - timestamp) < 0.1 for sc in scene_changes)
                scene_type = SceneType.TRANSITION if is_scene_change else SceneType.STATIC

                if prev_timestamp >= 0:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx - interval)
                    ret, prev_frame = cap.read()
                    if ret:
                        prev_frame_rgb = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2RGB)
                        gray1 = cv2.cvtColor(prev_frame_rgb, cv2.COLOR_RGB2GRAY)
                        gray2 = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
                        diff = cv2.absdiff(gray1, gray2)
                        diff_score = np.mean(diff)
                    else:
                        diff_score = 0.0
                else:
                    diff_score = 0.0

                frames.append(Frame(
                    image=frame_rgb,
                    timestamp=timestamp,
                    scene_type=scene_type,
                    difference_score=diff_score
                ))

                prev_timestamp = timestamp

            self.logger.info(f"Selected {len(frames)} frames")
            return frames

        finally:
            cap.release()

class UniformFrameSelector(FrameSelector):
    """Selects frames at uniform intervals without scene detection"""

    def select_frames(self, video_path: str, min_frames: int, max_frames: int,
                      frames_per_minute: float) -> List[Frame]:
        """Select frames at uniform intervals"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video file: {video_path}")

        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps

            # Calculate number of frames to extract
            target_frames = min(max_frames, max(min_frames,
                                                int(duration / 60 * frames_per_minute)))

            self.logger.info(f"Selecting {target_frames} frames uniformly from video...")

            # Extract frames at uniform intervals
            frames = []
            interval = total_frames // target_frames

            for frame_idx in range(0, total_frames, interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    break

                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                frames.append(Frame(
                    image=frame_rgb,
                    timestamp=timestamp,
                    scene_type=SceneType.STATIC
                ))

            return frames

        finally:
            cap.release()

