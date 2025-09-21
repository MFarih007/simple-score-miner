"""Configuration settings for Simple Miner."""

import os
from pathlib import Path

# Directories
PROJECT_ROOT = Path(__file__).parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
TEMP_DIR = PROJECT_ROOT / "temp"

# Ensure directories exist
MODELS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

# Server Configuration
HOST = os.getenv("MINER_HOST", "0.0.0.0")
PORT = int(os.getenv("MINER_PORT", "8000"))

# Model Configuration
PLAYER_MODEL_NAME = "yolov8n.pt"  # You can use custom models here
KEYPOINT_MODEL_NAME = "yolov8n-pose.pt"  # For pose/keypoint detection

# Processing Configuration
MAX_PROCESSING_TIME = 300  # 5 minutes timeout
FRAME_PROCESSING_BATCH_SIZE = 10
MAX_VIDEO_SIZE_MB = 500

# Device Configuration
DEVICE = os.getenv("DEVICE", "auto")  # auto, cpu, cuda, mps

# Soccer Field Configuration - Standard 16 keypoints
# These represent key points on a soccer field: corners, goals, center, etc.
SOCCER_KEYPOINTS = [
    "top_left_corner", "top_right_corner", "bottom_left_corner", "bottom_right_corner",
    "left_goal_top_left", "left_goal_top_right", "left_goal_bottom_left", "left_goal_bottom_right",
    "right_goal_top_left", "right_goal_top_right", "right_goal_bottom_left", "right_goal_bottom_right",
    "center_top", "center_bottom", "center_left", "center_right"
]

# Class IDs for object detection
CLASS_NAMES = {
    0: "ball",
    1: "goalkeeper", 
    2: "player",
    3: "referee"
}

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")