"""Core miner implementation for soccer video analysis."""

import time
import torch
import asyncio
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from loguru import logger

# Import AI libraries
from ultralytics import YOLO
import supervision as sv
import cv2

from config import (
    MODELS_DIR, 
    PLAYER_MODEL_NAME, 
    KEYPOINT_MODEL_NAME, 
    DEVICE,
    SOCCER_KEYPOINTS,
    CLASS_NAMES
)
from .video_utils import VideoDownloader, VideoProcessor


class DeviceManager:
    """Manages device selection for AI models."""
    
    @staticmethod
    def get_optimal_device() -> str:
        """Determine the best available device for AI processing."""
        if DEVICE != "auto":
            return DEVICE
            
        # Check for CUDA
        if torch.cuda.is_available():
            device = "cuda"
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
        # Check for MPS (Apple Silicon)
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
            logger.info("Using Apple Silicon MPS device")
        else:
            device = "cpu"
            logger.info("Using CPU device")
            
        return device


class ModelManager:
    """Manages AI model loading and inference."""
    
    def __init__(self, device: str = "auto"):
        self.device = DeviceManager.get_optimal_device() if device == "auto" else device
        self.models = {}
        self.tracker = None
        
    def load_models(self):
        """Load all required AI models."""
        try:
            logger.info("Loading AI models...")
            
            # Load player detection model
            player_model_path = MODELS_DIR / PLAYER_MODEL_NAME
            if not player_model_path.exists():
                logger.info(f"Downloading {PLAYER_MODEL_NAME}...")
                self.models['player'] = YOLO(PLAYER_MODEL_NAME)
                # Save the model locally for future use
                # self.models['player'].save(str(player_model_path))
            else:
                self.models['player'] = YOLO(str(player_model_path))
            
            # Load keypoint detection model
            keypoint_model_path = MODELS_DIR / KEYPOINT_MODEL_NAME
            if not keypoint_model_path.exists():
                logger.info(f"Downloading {KEYPOINT_MODEL_NAME}...")
                self.models['keypoints'] = YOLO(KEYPOINT_MODEL_NAME)
                # self.models['keypoints'].save(str(keypoint_model_path))
            else:
                self.models['keypoints'] = YOLO(str(keypoint_model_path))
            
            # Initialize object tracker
            self.tracker = sv.ByteTrack()
            
            logger.info(f"Models loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load models: {str(e)}")
            raise
    
    def detect_objects(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Detect players, ball, referees, and goalkeepers in frame.
        
        Args:
            frame: Input video frame
            
        Returns:
            Dictionary containing detection results
        """
        try:
            # Run inference
            results = self.models['player'](frame, verbose=False)[0]
            
            # Convert to supervision format
            detections = sv.Detections.from_ultralytics(results)
            
            # Update tracker
            if self.tracker:
                detections = self.tracker.update_with_detections(detections)
            
            # Format results
            objects = []
            if detections.xyxy is not None and len(detections.xyxy) > 0:
                for i in range(len(detections.xyxy)):
                    bbox = detections.xyxy[i].tolist()  # [x1, y1, x2, y2]
                    class_id = int(detections.class_id[i]) if detections.class_id is not None else 2  # Default to player
                    track_id = int(detections.tracker_id[i]) if detections.tracker_id is not None else i
                    confidence = float(detections.confidence[i]) if detections.confidence is not None else 0.5
                    
                    # Map class_id to our soccer classes (simplified mapping)
                    # You would need to train a custom model for accurate soccer object detection
                    if class_id == 0:  # person class from COCO -> player
                        soccer_class_id = 2  # player
                    else:
                        soccer_class_id = class_id
                    
                    objects.append({
                        "id": track_id,
                        "bbox": [float(x) for x in bbox],  # Ensure native Python floats
                        "class_id": soccer_class_id,
                        "confidence": confidence
                    })
            
            return {"objects": objects}
            
        except Exception as e:
            logger.error(f"Object detection failed: {str(e)}")
            return {"objects": []}
    
    def detect_keypoints(self, frame: np.ndarray) -> List[List[float]]:
        """
        Detect soccer field keypoints in frame.
        
        Args:
            frame: Input video frame
            
        Returns:
            List of 16 keypoint coordinates [[x, y], ...]
        """
        try:
            # For this simplified version, we'll use pose detection as a proxy
            # In a real implementation, you'd need a specialized soccer field keypoint model
            results = self.models['keypoints'](frame, verbose=False)[0]
            
            # Initialize 16 soccer keypoints (all zeros if not detected)
            keypoints = [[0.0, 0.0] for _ in range(16)]
            
            # Extract keypoints from pose detection results
            if hasattr(results, 'keypoints') and results.keypoints is not None:
                kpts = results.keypoints.xy[0] if len(results.keypoints.xy) > 0 else None
                if kpts is not None and len(kpts) > 0:
                    # Map some pose keypoints to soccer field keypoints (simplified)
                    # This is a placeholder - you'd need proper field detection
                    for i, kpt in enumerate(kpts[:min(16, len(kpts))]):
                        if not torch.isnan(kpt).any():
                            keypoints[i] = [float(kpt[0]), float(kpt[1])]
            
            # If no keypoints detected, try to detect field corners/lines using computer vision
            if all(kp == [0.0, 0.0] for kp in keypoints):
                cv_keypoints = self._detect_field_keypoints_cv(frame)
                if cv_keypoints:
                    keypoints = cv_keypoints
            
            return keypoints
            
        except Exception as e:
            logger.error(f"Keypoint detection failed: {str(e)}")
            return [[0.0, 0.0] for _ in range(16)]
    
    def _detect_field_keypoints_cv(self, frame: np.ndarray) -> Optional[List[List[float]]]:
        """
        Use computer vision techniques to detect field corners and lines.
        This is a simplified implementation for demonstration.
        """
        try:
            # Convert to HSV to detect grass
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Create mask for green grass
            lower_green = np.array([35, 40, 40])
            upper_green = np.array([85, 255, 255])
            grass_mask = cv2.inRange(hsv, lower_green, upper_green)
            
            # Find contours
            contours, _ = cv2.findContours(grass_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find the largest contour (likely the field)
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # Generate some basic keypoints based on field boundaries
                keypoints = [[0.0, 0.0] for _ in range(16)]
                
                # Corner points
                keypoints[0] = [float(x), float(y)]  # top-left
                keypoints[1] = [float(x + w), float(y)]  # top-right
                keypoints[2] = [float(x), float(y + h)]  # bottom-left
                keypoints[3] = [float(x + w), float(y + h)]  # bottom-right
                
                # Center points
                keypoints[12] = [float(x + w//2), float(y)]  # center-top
                keypoints[13] = [float(x + w//2), float(y + h)]  # center-bottom
                keypoints[14] = [float(x), float(y + h//2)]  # center-left
                keypoints[15] = [float(x + w), float(y + h//2)]  # center-right
                
                return keypoints
                
        except Exception as e:
            logger.error(f"Computer vision keypoint detection failed: {str(e)}")
        
        return None


class SimpleMiner:
    """Main miner class that processes soccer video challenges."""
    
    def __init__(self):
        self.model_manager = ModelManager()
        self.video_downloader = VideoDownloader()
        self.video_processor = VideoProcessor()
        self.is_initialized = False
        
    async def initialize(self):
        """Initialize the miner with all required components."""
        try:
            logger.info("Initializing Simple Miner...")
            
            # Load AI models
            self.model_manager.load_models()
            
            self.is_initialized = True
            logger.info("Simple Miner initialized successfully!")
            
        except Exception as e:
            logger.error(f"Miner initialization failed: {str(e)}")
            raise
    
    async def process_challenge(self, challenge_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a soccer video challenge.
        
        Args:
            challenge_data: Challenge information containing video_url, challenge_id, etc.
            
        Returns:
            Dictionary containing processed results
        """
        if not self.is_initialized:
            await self.initialize()
        
        start_time = time.time()
        challenge_id = challenge_data.get("challenge_id", "unknown")
        video_url = challenge_data.get("video_url")
        
        logger.info(f"Processing challenge {challenge_id}")
        
        if not video_url:
            raise ValueError("No video URL provided in challenge")
        
        video_path = None
        try:
            # Download video
            video_path = await self.video_downloader.download_video(video_url)
            if not video_path:
                raise ValueError("Failed to download video")
            
            # Validate video
            if not await self.video_processor.validate_video(video_path):
                raise ValueError("Invalid video file")
            
            # Get video info
            video_info = await self.video_processor.get_video_info(video_path)
            logger.info(f"Processing video: {video_info}")
            
            # Process frames
            frames = []
            async for frame_number, frame in self.video_processor.stream_frames(video_path):
                # Detect objects
                object_results = self.model_manager.detect_objects(frame)
                
                # Detect keypoints
                keypoints = self.model_manager.detect_keypoints(frame)
                
                # Create frame data
                frame_data = {
                    "frame_number": int(frame_number),
                    "keypoints": keypoints,
                    "objects": object_results["objects"]
                }
                
                frames.append(frame_data)
                
                # Log progress every 50 frames
                if frame_number % 50 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_number / elapsed if elapsed > 0 else 0
                    logger.info(f"Processed {frame_number} frames ({fps:.2f} fps)")
            
            processing_time = time.time() - start_time
            
            response = {
                "challenge_id": challenge_id,
                "frames": frames,
                "processing_time": processing_time,
                "video_info": video_info,
                "total_frames": len(frames)
            }
            
            logger.info(f"Challenge {challenge_id} completed in {processing_time:.2f}s")
            logger.info(f"Processed {len(frames)} frames ({len(frames)/processing_time:.2f} fps)")
            
            return response
            
        except Exception as e:
            logger.error(f"Challenge processing failed: {str(e)}")
            raise
        finally:
            # Cleanup
            if video_path:
                self.video_downloader.cleanup_video(video_path)