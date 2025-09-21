# Simple Soccer Miner

A simplified implementation of the Score Vision miner that processes soccer videos to detect players and field keypoints, similar to the functionality in the Score Vision (SN44) Bittensor subnet.

## 🚀 Features

- **Player Detection**: Detects players, goalkeepers, referees, and ball in video frames
- **Object Tracking**: Maintains consistent tracking IDs across frames using ByteTrack
- **Keypoint Detection**: Extracts soccer field keypoints (corners, goals, center points)
- **FastAPI Server**: RESTful API for processing video challenges
- **Video Processing**: Downloads and processes videos from URLs
- **Real-time Processing**: Streams video frames for efficient processing

## 📋 Requirements

- Python 3.8+
- PyTorch (for AI models)
- OpenCV (for video processing)
- FastAPI (for web server)
- Internet connection (for downloading models and videos)

### Hardware Requirements

- **Minimum**: CPU-only processing (slower)
- **Recommended**: NVIDIA GPU with CUDA support
- **Apple Silicon**: Supports MPS acceleration
- **Memory**: 4GB+ RAM, 2GB+ free disk space for models

## 🛠️ Installation

### 1. Clone or Download

```bash
# If you have the project folder
cd /Users/mini/proj/simple-miner

# Or create from scratch
mkdir simple-miner && cd simple-miner
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install PyTorch (if not automatically installed)

For CPU-only:
```bash
pip install torch torchvision
```

For CUDA (NVIDIA GPU):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## 🏃‍♂️ Quick Start

### 1. Start the Server

```bash
python -m src.main
```

The server will start on `http://localhost:8000` by default.

### 2. Test the API

Open another terminal and test:

```bash
# Test health endpoint
curl http://localhost:8000/health

# Test info endpoint
curl http://localhost:8000/info
```

### 3. Process a Video Challenge

```bash
curl -X POST "http://localhost:8000/soccer/challenge" \
  -H "Content-Type: application/json" \
  -d '{
    "challenge_id": "test_001",
    "video_url": "https://sample-videos.com/zip/10/mp4/SampleVideo_720x480_1mb.mp4",
    "type": "gsr"
  }'
```

## 🧪 Running Tests

Run the built-in test suite:

```bash
# Run tests (server should be running in another terminal)
python tests/test_miner.py
```

This will test:
- Direct miner functionality
- API endpoints
- Video processing pipeline
- Model loading and inference

## 📚 API Documentation

### Endpoints

#### `GET /` - Root Information
Returns basic server information and available endpoints.

#### `GET /health` - Health Check
```json
{
  "status": "healthy",
  "message": "Simple Miner is running", 
  "initialized": true
}
```

#### `GET /info` - Miner Information
Returns detailed information about miner capabilities and models.

#### `POST /soccer/challenge` - Process Challenge
Process a soccer video challenge.

**Request:**
```json
{
  "challenge_id": "unique_id",
  "video_url": "https://example.com/video.mp4",
  "type": "gsr"
}
```

**Response:**
```json
{
  "challenge_id": "unique_id",
  "frames": [
    {
      "frame_number": 0,
      "keypoints": [[x1, y1], [x2, y2], ...],
      "objects": [
        {
          "id": 1,
          "bbox": [x1, y1, x2, y2],
          "class_id": 2,
          "confidence": 0.85
        }
      ]
    }
  ],
  "processing_time": 45.2,
  "total_frames": 300
}
```

## 🏗️ Architecture

```
simple-miner/
├── src/
│   ├── __init__.py
│   ├── main.py          # Main entry point
│   ├── server.py        # FastAPI server
│   ├── miner.py         # Core miner logic
│   └── video_utils.py   # Video processing utilities
├── models/              # AI models (auto-downloaded)
├── data/                # Data storage
├── temp/                # Temporary files
├── tests/
│   └── test_miner.py    # Test suite
├── config.py            # Configuration
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

### Core Components

1. **SimpleMiner**: Main processing class
2. **ModelManager**: Handles AI model loading and inference
3. **VideoDownloader**: Downloads videos from URLs
4. **VideoProcessor**: Streams and processes video frames
5. **FastAPI Server**: Web API for receiving challenges

## ⚙️ Configuration

Edit `config.py` to customize:

```python
# Server settings
HOST = "0.0.0.0"
PORT = 8000

# Device selection
DEVICE = "auto"  # auto, cpu, cuda, mps

# Processing settings
MAX_PROCESSING_TIME = 300
FRAME_PROCESSING_BATCH_SIZE = 10
```

Environment variables:
- `MINER_HOST`: Server host (default: 0.0.0.0)
- `MINER_PORT`: Server port (default: 8000)
- `DEVICE`: Processing device (default: auto)
- `LOG_LEVEL`: Logging level (default: INFO)

## 🔧 How It Works

### 1. Video Processing Pipeline

1. **Download**: Video downloaded from provided URL
2. **Validation**: Check video format and readability
3. **Frame Streaming**: Process frames sequentially
4. **AI Inference**: Run object detection and keypoint detection
5. **Tracking**: Maintain object IDs across frames
6. **Response**: Format and return results

### 2. AI Models

- **Player Detection**: YOLOv8n for detecting people (mapped to players)
- **Keypoint Detection**: YOLOv8n-pose + computer vision for field keypoints
- **Tracking**: ByteTrack for maintaining consistent object IDs

### 3. Object Classes

- `0`: Ball
- `1`: Goalkeeper
- `2`: Player
- `3`: Referee

### 4. Keypoint Structure

16 soccer field keypoints:
- Corners (4 points)
- Goal posts (8 points)
- Center points (4 points)

## 🐛 Troubleshooting

### Common Issues

1. **Model Download Fails**
   - Check internet connection
   - Ensure sufficient disk space
   - Try running again (models auto-download)

2. **CUDA Errors**
   - Install correct PyTorch version for your CUDA version
   - Check GPU memory availability
   - Fall back to CPU: `DEVICE=cpu python -m src.main`

3. **Video Download Fails**
   - Check video URL accessibility
   - Verify video format support
   - Check network connectivity

4. **Import Errors**
   - Ensure virtual environment is activated
   - Install dependencies: `pip install -r requirements.txt`
   - Check Python version (3.8+)

### Performance Tips

- Use GPU acceleration for better performance
- Reduce video resolution for faster processing
- Process shorter video clips for testing
- Monitor system resources during processing

## 🔄 Comparison with Score Vision

This simplified miner demonstrates the core concepts of the Score Vision miner:

### ✅ Similar Features
- Video processing pipeline
- Object detection and tracking
- FastAPI server architecture
- Challenge/response format
- AI model integration

### 📝 Simplifications
- Uses generic YOLOv8 models instead of specialized soccer models
- Simplified keypoint detection using computer vision
- Basic tracking without advanced features
- No Bittensor integration
- No cryptographic verification
- Simplified error handling

### 🚀 Potential Improvements
- Train custom soccer-specific models
- Implement proper soccer field keypoint detection
- Add more sophisticated tracking algorithms
- Integrate with Bittensor network
- Add cryptographic verification
- Implement advanced evaluation metrics

## 📄 License

MIT License - Feel free to use and modify for educational purposes.

## 🤝 Contributing

This is a educational/demonstration project. Feel free to:
- Report bugs
- Suggest improvements
- Add features
- Create pull requests

## 📞 Support

For questions or issues:
1. Check the troubleshooting section
2. Run the test suite to identify issues
3. Check server logs for detailed error messages
4. Ensure all dependencies are properly installed