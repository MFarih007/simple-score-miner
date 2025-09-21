"""Test script for Simple Miner functionality."""

import asyncio
import json
import httpx
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.miner import SimpleMiner
from config import HOST, PORT


async def test_miner_direct():
    """Test the miner functionality directly (without API)."""
    print("=" * 50)
    print("Testing Simple Miner directly...")
    print("=" * 50)
    
    try:
        # Initialize miner
        miner = SimpleMiner()
        await miner.initialize()
        print("✅ Miner initialized successfully")
        
        # Test with a sample video URL (you can replace this)
        test_challenge = {
            "challenge_id": "direct_test_001",
            # "video_url": "https://sample-videos.com/zip/10/mp4/SampleVideo_720x480_1mb.mp4",
            "video_url": "https://media.istockphoto.com/id/1426881162/video/soccer-football-championship-stadium-with-crowd-of-fans-blue-team-attack-and-score-goal.mp4?s=mp4-640x640-is&k=20&c=5uaH1mM-5nF9n6dqQmo7-Wp4n21EaUqnVanyH9Th4Rs=",
            "type": "gsr"
        }
        
        print(f"📹 Processing test video: {test_challenge['video_url']}")
        
        # Process challenge
        result = await miner.process_challenge(test_challenge)
        
        # Display results
        print("\n📊 Processing Results:")
        print(f"  Challenge ID: {result['challenge_id']}")
        print(f"  Total frames: {result['total_frames']}")
        print(f"  Processing time: {result['processing_time']:.2f} seconds")
        print(f"  Average FPS: {result['total_frames']/result['processing_time']:.2f}")
        
        if result['video_info']:
            print(f"  Video info: {result['video_info']}")
        
        # Show sample frame data
        if result['frames']:
            sample_frame = result['frames'][0]
            print(f"\n🎬 Sample frame data:")
            print(f"  Frame number: {sample_frame['frame_number']}")
            print(f"  Objects detected: {len(sample_frame['objects'])}")
            print(f"  Keypoints: {len([kp for kp in sample_frame['keypoints'] if kp != [0.0, 0.0]])} valid out of 16")
            
            if sample_frame['objects']:
                print(f"  Sample objects: {sample_frame['objects'][:3]}...")
        
        print("✅ Direct test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Direct test failed: {str(e)}")
        return False


async def test_miner_api():
    """Test the miner via API endpoints."""
    print("\n" + "=" * 50)
    print("Testing Simple Miner API...")
    print("=" * 50)
    
    base_url = f"http://{HOST}:{PORT}"
    
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            # Test health endpoint
            print("🩺 Testing health endpoint...")
            health_response = await client.get(f"{base_url}/health")
            health_data = health_response.json()
            print(f"  Status: {health_data['status']}")
            print(f"  Message: {health_data['message']}")
            print(f"  Initialized: {health_data['initialized']}")
            
            # Test info endpoint
            print("\n📋 Testing info endpoint...")
            info_response = await client.get(f"{base_url}/info")
            info_data = info_response.json()
            print(f"  Name: {info_data['name']}")
            print(f"  Capabilities: {len(info_data['capabilities'])} listed")
            print(f"  Models: {info_data['models']}")
            
            # Test challenge endpoint
            print("\n⚽ Testing challenge endpoint...")
            challenge_data = {
                "challenge_id": "api_test_001",
                "video_url": "https://sample-videos.com/zip/10/mp4/SampleVideo_720x480_1mb.mp4",
                "type": "gsr"
            }
            
            print(f"📤 Sending challenge: {challenge_data['challenge_id']}")
            challenge_response = await client.post(
                f"{base_url}/soccer/challenge",
                json=challenge_data
            )
            
            if challenge_response.status_code == 200:
                result = challenge_response.json()
                print("✅ Challenge processed successfully!")
                print(f"  Challenge ID: {result['challenge_id']}")
                print(f"  Total frames: {result['total_frames']}")
                print(f"  Processing time: {result['processing_time']:.2f} seconds")
                return True
            else:
                print(f"❌ Challenge failed with status: {challenge_response.status_code}")
                print(f"  Error: {challenge_response.text}")
                return False
                
    except httpx.ConnectError:
        print("❌ Could not connect to API server. Make sure the server is running.")
        print(f"   Try running: python -m src.main")
        return False
    except Exception as e:
        print(f"❌ API test failed: {str(e)}")
        return False


async def run_tests():
    """Run all tests."""
    print("🚀 Starting Simple Miner Tests")
    print("=" * 60)
    
    # Test direct miner functionality
    direct_success = await test_miner_direct()
    
    # Test API (only if server is running)
    api_success = await test_miner_api()
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 Test Summary:")
    print(f"  Direct Miner Test: {'✅ PASSED' if direct_success else '❌ FAILED'}")
    print(f"  API Test: {'✅ PASSED' if api_success else '❌ FAILED'}")
    
    if direct_success and api_success:
        print("\n🎉 All tests passed!")
    elif direct_success:
        print("\n⚠️  Direct test passed, but API test failed.")
        print("   Make sure to start the server with: python -m src.main")
    else:
        print("\n❌ Tests failed. Check the error messages above.")


def main():
    """Main test function."""
    print("Simple Miner Test Suite")
    print("This will test the miner functionality and API endpoints.")
    print("\nNote: Make sure you have internet connection for video download tests.")
    
    try:
        asyncio.run(run_tests())
    except KeyboardInterrupt:
        print("\n⏹️  Tests interrupted by user")
    except Exception as e:
        print(f"\n💥 Test suite crashed: {str(e)}")


if __name__ == "__main__":
    main()