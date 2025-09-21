"""Main entry point for Simple Miner."""

import sys
import asyncio
from pathlib import Path
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import LOG_LEVEL
from src.server import main as server_main


def setup_logging():
    """Configure logging settings."""
    # Remove default handler
    logger.remove()
    
    # Add custom handler with formatting
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=LOG_LEVEL
    )


def main():
    """Main entry point."""
    try:
        setup_logging()
        logger.info("Starting Simple Miner application...")
        
        # Run the FastAPI server
        server_main()
        
    except KeyboardInterrupt:
        logger.info("Shutting down Simple Miner...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()