import cv2
import numpy as np
import os

def test_opencv():
    """Test if OpenCV is working correctly"""
    print("Testing OpenCV installation...")
    print(f"OpenCV version: {cv2.__version__}")
    
    # Test 1: Create a simple image and save it
    print("\nTest 1: Creating and saving a test image...")
    test_image = np.zeros((100, 100, 3), dtype=np.uint8)
    test_image[:, :] = [0, 255, 0]  # Green image
    
    test_path = "test_image.jpg"
    success = cv2.imwrite(test_path, test_image)
    
    if success and os.path.exists(test_path):
        file_size = os.path.getsize(test_path)
        print(f"✓ Successfully created test image: {test_path} ({file_size} bytes)")
        os.remove(test_path)  # Clean up
    else:
        print("✗ Failed to create test image")
        return False
    
    print("\nOpenCV is working correctly!")
    return True

if __name__ == "__main__":
    test_opencv()
