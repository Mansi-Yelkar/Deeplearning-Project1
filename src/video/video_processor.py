from pathlib import Path
from typing import Generator, Tuple, Dict, Any, Optional
import cv2
import numpy as np

from src.utils.logging_config import setup_logger

logger = setup_logger("video_processor")

class VideoProcessor:
    """Streams video frame-by-frame incrementally to preserve memory."""

    def __init__(self, video_path: str, sample_rate: int = 5):
        self.video_path = video_path
        self.sample_rate = max(1, sample_rate)
        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration_seconds = self.total_frames / self.fps if self.fps > 0 else 0.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(
            f"Initialized video reader: {video_path} "
            f"({self.width}x{self.height}, {self.fps:.1f} fps, {self.total_frames} frames, {self.duration_seconds:.2f}s)"
        )

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "fps": round(self.fps, 2),
            "total_frames": self.total_frames,
            "duration_seconds": round(self.duration_seconds, 2),
            "width": self.width,
            "height": self.height,
            "sample_rate": self.sample_rate
        }

    def frame_generator(self) -> Generator[Tuple[int, float, np.ndarray], None, None]:
        """Yields (frame_index, timestamp_seconds, frame_bgr) for every sampled frame."""
        frame_idx = 0
        try:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break

                if frame_idx % self.sample_rate == 0:
                    timestamp_sec = frame_idx / self.fps if self.fps > 0 else 0.0
                    yield frame_idx, round(timestamp_sec, 2), frame

                frame_idx += 1
        finally:
            self.release()

    def release(self) -> None:
        if self.cap and self.cap.isOpened():
            self.cap.release()
