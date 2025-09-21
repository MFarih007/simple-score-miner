from setuptools import setup, find_packages

setup(
    name="simple-miner",
    version="1.0.0",
    description="A simplified implementation of Score Vision miner for soccer video analysis",
    author="Simple Miner Project",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.104.1",
        "uvicorn>=0.24.0",
        "httpx>=0.25.0",
        "python-multipart>=0.0.6",
        "opencv-python>=4.8.0",
        "ultralytics>=8.0.196",
        "supervision>=0.15.0",
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "ffmpeg-python>=0.2.0",
        "numpy>=1.24.0",
        "pillow>=10.0.0",
        "pydantic>=2.4.0",
        "loguru>=0.7.2",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "simple-miner=src.main:main",
        ],
    },
)