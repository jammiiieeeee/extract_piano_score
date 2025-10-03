import cv2
import numpy as np
import os

def create_test_video():
    """Create a simple test video for debugging"""
    print("Creating test video...")
    
    # Video properties
    width, height = 640, 480
    fps = 30
    duration = 30  # 30 seconds
    total_frames = fps * duration
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('test_video.mp4', fourcc, fps, (width, height))
    
    if not out.isOpened():
        print("Error: Could not create video writer")
        return False
    
    # Generate frames
    for frame_num in range(total_frames):
        # Create a frame with changing colors and text
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Change background color over time
        color_value = int((frame_num / total_frames) * 255)
        frame[:, :] = [color_value, 255 - color_value, 128]
        
        # Add frame number text
        seconds = frame_num / fps
        text = f"Frame {frame_num} - Time: {seconds:.1f}s"
        cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Add a moving circle
        center_x = int(width * 0.5 + 200 * np.sin(frame_num * 0.1))
        center_y = int(height * 0.5)
        cv2.circle(frame, (center_x, center_y), 50, (0, 255, 255), -1)
        
        out.write(frame)
        
        if frame_num % 60 == 0:  # Print progress every 2 seconds
            print(f"Generated {frame_num}/{total_frames} frames...")
    
    out.release()
    
    if os.path.exists('test_video.mp4'):
        file_size = os.path.getsize('test_video.mp4')
        print(f"✓ Test video created: test_video.mp4 ({file_size:,} bytes)")
        return True
    else:
        print("✗ Failed to create test video")
        return False

if __name__ == "__main__":
    create_test_video()
