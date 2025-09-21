"""Video processing utilities for the Simple Miner."""

import os
import cv2
import time
import httpx
import asyncio
from pathlib import Path
from typing import Generator, Tuple, Optional
from loguru import logger
import tempfile
import hashlib

from config import TEMP_DIR, MAX_VIDEO_SIZE_MB


class VideoDownloader:
    """Handles video download and management."""
    
    def __init__(self):
        self.temp_dir = TEMP_DIR
        self.temp_dir.mkdir(exist_ok=True)
    
    async def download_video(self, video_url: str, timeout: float = 120.0) -> Optional[Path]:
        """
        Download video from URL and save to temp directory.
        
        Args:
            video_url: URL of the video to download
            timeout: Download timeout in seconds
            
        Returns:
            Path to downloaded video file, or None if failed
        """
        try:
            # Create unique filename based on URL hash
            url_hash = hashlib.md5(video_url.encode()).hexdigest()[:8]
            timestamp = int(time.time())
            video_path = self.temp_dir / f"video_{url_hash}_{timestamp}.mp4"
            
            logger.info(f"Downloading video from {video_url}")
            
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream("GET", video_url) as response:
                    response.raise_for_status()
                    
                    # Check content length
                    content_length = response.headers.get("content-length")
                    if content_length:
                        size_mb = int(content_length) / (1024 * 1024)
                        if size_mb > MAX_VIDEO_SIZE_MB:
                            logger.warning(f"Video too large: {size_mb:.1f}MB > {MAX_VIDEO_SIZE_MB}MB")
                            return None
                    
                    # Download the video
                    with open(video_path, "wb") as f:
                        async for chunk in response.aiter_bytes(chunk_size=8192):
                            f.write(chunk)
            
            logger.info(f"Video downloaded successfully: {video_path}")
            return video_path
            
        except Exception as e:
            logger.error(f"Failed to download video: {str(e)}")
            return None
    
    def cleanup_video(self, video_path: Path) -> None:
        """Remove downloaded video file."""
        try:
            if video_path.exists():
                video_path.unlink()
                logger.info(f"Cleaned up video: {video_path}")
        except Exception as e:
            logger.warning(f"Failed to cleanup video {video_path}: {str(e)}")


class VideoProcessor:
    """Handles video frame processing and streaming."""
    
    def __init__(self, batch_size: int = 10):
        self.batch_size = batch_size
    
    async def get_video_info(self, video_path: Path) -> dict:
        """Get basic information about the video."""
        try:
            cap = cv2.VideoCapture(str(video_path))
            
            if not cap.isOpened():
                raise ValueError("Cannot open video file")
            
            info = {
                "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                "fps": cap.get(cv2.CAP_PROP_FPS),
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "duration": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS))
            }
            
            cap.release()
            return info
            
        except Exception as e:
            logger.error(f"Failed to get video info: {str(e)}")
            return {}
    
    async def stream_frames(self, video_path: Path) -> Generator[Tuple[int, any], None, None]:
        """
        Stream frames from video file.
        
        Args:
            video_path: Path to video file
            
        Yields:
            Tuple of (frame_number, frame_data)
        """
        try:
            cap = cv2.VideoCapture(str(video_path))
            
            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {video_path}")
            
            frame_number = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                yield frame_number, frame
                frame_number += 1
                
                # Yield control periodically to allow other async tasks
                if frame_number % self.batch_size == 0:
                    await asyncio.sleep(0.001)  # Small delay to yield control
            
            cap.release()
            logger.info(f"Processed {frame_number} frames from {video_path}")
            
        except Exception as e:
            logger.error(f"Error streaming frames: {str(e)}")
            raise
    
    async def validate_video(self, video_path: Path) -> bool:
        """Validate that video file is readable and processable."""
        try:
            if not video_path.exists():
                logger.error(f"Video file does not exist: {video_path}")
                return False
            
            # Try to open and read first frame
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.error(f"Cannot open video file: {video_path}")
                return False
            
            ret, frame = cap.read()
            cap.release()
            
            if not ret or frame is None:
                logger.error(f"Cannot read frames from video: {video_path}")
                return False
            
            # Check frame dimensions
            height, width = frame.shape[:2]
            if height < 240 or width < 320:
                logger.error(f"Video resolution too small: {width}x{height}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Video validation failed: {str(e)}")
            return False