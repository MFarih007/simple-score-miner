"""FastAPI server for Simple Miner."""

import asyncio
from typing import Dict, Any
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from pydantic import BaseModel
from loguru import logger
import uvicorn

from .miner import SimpleMiner
from config import HOST, PORT


# Pydantic models for request/response validation
class ChallengeRequest(BaseModel):
    """Challenge request format."""
    challenge_id: str
    video_url: str
    type: str = "gsr"  # Game State Recognition
    created_at: str = None


class ChallengeResponse(BaseModel):
    """Challenge response format."""
    challenge_id: str
    frames: list
    processing_time: float
    video_info: dict = None
    total_frames: int = 0


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    message: str
    initialized: bool


# Global miner instance
miner_instance = None
processing_lock = asyncio.Lock()


def get_miner() -> SimpleMiner:
    """Get or create the global miner instance."""
    global miner_instance
    if miner_instance is None:
        miner_instance = SimpleMiner()
    return miner_instance


# Create FastAPI app
app = FastAPI(
    title="Simple Soccer Miner",
    description="A simplified implementation of Score Vision miner for soccer video analysis",
    version="1.0.0"
)


@app.on_event("startup")
async def startup_event():
    """Initialize the miner on startup."""
    try:
        logger.info("Starting Simple Miner server...")
        miner = get_miner()
        await miner.initialize()
        logger.info("Simple Miner server started successfully!")
    except Exception as e:
        logger.error(f"Failed to initialize miner: {str(e)}")
        raise


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        miner = get_miner()
        return HealthResponse(
            status="healthy",
            message="Simple Miner is running",
            initialized=miner.is_initialized
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthResponse(
            status="unhealthy", 
            message=f"Error: {str(e)}",
            initialized=False
        )


@app.get("/")
async def root():
    """Root endpoint with basic information."""
    return {
        "name": "Simple Soccer Miner",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "challenge": "/soccer/challenge",
            "info": "/info"
        }
    }


@app.get("/info")
async def get_info():
    """Get miner information and capabilities."""
    return {
        "name": "Simple Soccer Miner",
        "description": "Processes soccer videos to detect players and field keypoints",
        "capabilities": [
            "Player detection and tracking",
            "Soccer field keypoint detection", 
            "Object classification (players, ball, referees, goalkeepers)",
            "Frame-by-frame video analysis"
        ],
        "supported_formats": ["mp4", "avi", "mov", "mkv"],
        "models": {
            "player_detection": "YOLOv8n",
            "keypoint_detection": "YOLOv8n-pose + Computer Vision",
            "tracking": "ByteTrack"
        }
    }


@app.post("/soccer/challenge", response_model=ChallengeResponse)
async def process_challenge(challenge: ChallengeRequest, background_tasks: BackgroundTasks):
    """
    Process a soccer video challenge.
    
    This endpoint receives a challenge with a video URL and processes it to extract:
    - Player detections and tracking
    - Soccer field keypoints
    - Frame-by-frame analysis
    """
    # Use lock to prevent concurrent processing (simplified approach)
    async with processing_lock:
        try:
            logger.info(f"Received challenge: {challenge.challenge_id}")
            
            miner = get_miner()
            if not miner.is_initialized:
                await miner.initialize()
            
            # Convert Pydantic model to dict for processing
            challenge_data = challenge.dict()
            
            # Process the challenge
            result = await miner.process_challenge(challenge_data)
            
            # Return response in expected format
            response = ChallengeResponse(
                challenge_id=result["challenge_id"],
                frames=result["frames"],
                processing_time=result["processing_time"],
                video_info=result.get("video_info", {}),
                total_frames=result.get("total_frames", len(result["frames"]))
            )
            
            logger.info(f"Challenge {challenge.challenge_id} completed successfully")
            return response
            
        except Exception as e:
            logger.error(f"Challenge processing failed: {str(e)}")
            raise HTTPException(
                status_code=500, 
                detail=f"Challenge processing failed: {str(e)}"
            )


@app.post("/test/challenge")
async def test_challenge():
    """
    Test endpoint that processes a sample challenge.
    Useful for testing the miner without needing an external video URL.
    """
    # This would use a test video URL - you can replace with any soccer video URL
    test_challenge_data = {
        "challenge_id": "test_001",
        "video_url": "https://media.istockphoto.com/id/1426881162/video/soccer-football-championship-stadium-with-crowd-of-fans-blue-team-attack-and-score-goal.mp4?s=mp4-640x640-is&k=20&c=5uaH1mM-5nF9n6dqQmo7-Wp4n21EaUqnVanyH9Th4Rs=",
        # "video_url": "https://sample-videos.com/zip/10/mp4/SampleVideo_720x480_1mb.mp4",
        "type": "gsr"
    }
    
    try:
        challenge = ChallengeRequest(**test_challenge_data)
        return await process_challenge(challenge, BackgroundTasks())
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Test challenge failed: {str(e)}"
        )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return HTTPException(
        status_code=500,
        detail=f"Internal server error: {str(exc)}"
    )


def main():
    """Main function to run the server."""
    logger.info(f"Starting Simple Miner server on {HOST}:{PORT}")
    uvicorn.run(
        "src.server:app",
        host=HOST,
        port=PORT,
        reload=False,  # Set to True for development
        log_level="info"
    )


if __name__ == "__main__":
    main()