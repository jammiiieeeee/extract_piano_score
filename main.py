import cv2
import os
import argparse
import sys
import re
from pathlib import Path

def sanitize_filename(filename):
    """
    Sanitize filename by removing or replacing problematic characters.
    """
    # Replace problematic characters with safe alternatives
    replacements = {
        '『': '[',
        '』': ']', 
        '／': '_',
        '：': '_',
        '？': '',
        '＜': '',
        '＞': '',
        '｜': '_',
        '＊': '',
        '"': '',
        "'": '',
        '\u3000': '_',  # Full-width space
    }
    
    # Apply replacements
    for old, new in replacements.items():
        filename = filename.replace(old, new)
    
    # Remove any remaining problematic characters using regex
    # Keep only alphanumeric, spaces, hyphens, underscores, periods, and brackets
    filename = re.sub(r'[^\w\s\-_.\[\]()]', '', filename)
    
    # Replace multiple spaces/underscores with single underscore
    filename = re.sub(r'[\s_]+', '_', filename)
    
    # Remove leading/trailing underscores and spaces
    filename = filename.strip('_ ')
    
    return filename

def detect_frame_change(frame1, frame2, top_ratio=0.2, threshold=0.04):
    """
    Detect if significant change occurred in the top portion of frames.
    
    Args:
        frame1: Previous frame (numpy array)
        frame2: Current frame (numpy array)
        top_ratio: Ratio of top portion to analyze (default: 0.2 = 20%)
        threshold: Threshold for considering change significant (default: 0.5 = 50%)
    
    Returns:
        bool: True if significant change detected
    """
    if frame1 is None or frame2 is None:
        return True
    
    # Get dimensions
    height, width = frame1.shape[:2]
    top_height = int(height * top_ratio)
    
    # Extract top portions
    top1 = frame1[0:top_height, 0:width]
    top2 = frame2[0:top_height, 0:width]
    
    # Convert to grayscale for comparison
    gray1 = cv2.cvtColor(top1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(top2, cv2.COLOR_BGR2GRAY)
    
    # Calculate absolute difference
    diff = cv2.absdiff(gray1, gray2)
    
    # Threshold the difference (pixels with change > 30 intensity levels)
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    
    # Calculate percentage of changed pixels
    total_pixels = top_height * width
    changed_pixels = cv2.countNonZero(thresh)
    change_percentage = changed_pixels / total_pixels
    
    return change_percentage >= threshold

def extract_screenshots(video_path, start_time=2, interval=12, detection_method='time', change_threshold=0.04):
    """
    Extract screenshots from a video at regular intervals or based on content changes.
    
    Args:
        video_path (str): Path to the video file
        start_time (int): Starting time in seconds (default: 2)
        interval (int): Interval between screenshots in seconds for time-based method (default: 12)
        detection_method (str): 'time' for fixed intervals, 'change' for content change detection
        change_threshold (float): Threshold for change detection (default: 0.5 = 50%)
    """
    # Check if video file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found.")
        return False
    
    # Get video filename without extension and sanitize it
    video_name = Path(video_path).stem
    sanitized_name = sanitize_filename(video_name)
    
    print(f"Original filename: {video_name}")
    print(f"Sanitized filename: {sanitized_name}")
    
    # Create organized folder structure:
    # [video_name]/
    #   ├── screenshots/
    #   └── [video_name]_score.pdf
    main_folder = os.path.join(os.getcwd(), sanitized_name)
    screenshots_dir = os.path.join(main_folder, "screenshots")
    
    # Delete existing folder if it exists
    if os.path.exists(main_folder):
        import shutil
        print(f"Existing folder found: {main_folder}")
        print("Deleting existing folder and contents...")
        shutil.rmtree(main_folder)
        print("✓ Existing folder deleted")
    
    # Create new folder structure
    os.makedirs(screenshots_dir, exist_ok=True)
    print(f"Created main folder: {main_folder}")
    print(f"Screenshots will be saved to: {screenshots_dir}")
    
    # Open video capture
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_path}'")
        return False
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"Video duration: {duration:.2f} seconds")
    print(f"Video FPS: {fps:.2f}")
    print(f"Total frames: {total_frames}")
    print(f"Screenshots will be saved to: {screenshots_dir}")
    
    # Check if we got valid video properties
    if fps <= 0 or total_frames <= 0:
        print("Error: Could not read video properties. The file might be corrupted or in an unsupported format.")
        cap.release()
        return False
    
    if duration <= start_time:
        print(f"Error: Video duration ({duration:.2f}s) is shorter than start time ({start_time}s)")
        cap.release()
        return False
    
    screenshot_count = 0
    
    # Create log file
    log_filename = f"{sanitized_name}_log.txt"
    log_path = os.path.join(main_folder, log_filename)
    
    if detection_method == 'time':
        print(f"Using time-based method: screenshots every {interval} seconds starting at {start_time}s")
        success = _extract_time_based(cap, fps, duration, start_time, interval, screenshots_dir, screenshot_count, log_path, sanitized_name)
    else:
        print(f"Using change-based method: detecting {change_threshold*100}% change in top 20% of frame")
        print(f"Log file will be saved to: {log_path}")
        success = _extract_change_based(cap, fps, duration, start_time, change_threshold, screenshots_dir, screenshot_count, log_path, sanitized_name)
    
    cap.release()
    return success, screenshots_dir

def _extract_time_based(cap, fps, duration, start_time, interval, screenshots_dir, screenshot_count, log_path, sanitized_name):
    """Extract screenshots at fixed time intervals"""
    import time
    
    current_time = start_time
    process_start_time = time.time()
    last_progress_update = 0
    
    # Initialize log file
    with open(log_path, 'w', encoding='utf-8') as log_file:
        log_file.write(f"Video Analysis Log - {sanitized_name}\n")
        log_file.write(f"Video Duration: {duration:.2f} seconds\n")
        log_file.write(f"Method: Time-based (every {interval}s)\n")
        log_file.write(f"Start Time: {start_time}s\n")
        log_file.write("=" * 50 + "\n\n")
    
    while current_time < duration:
        # Calculate frame number for current time
        frame_number = int(current_time * fps)
        
        # Set video position to the desired frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        # Read the frame
        ret, frame = cap.read()
        
        if ret:
            with open(log_path, 'a', encoding='utf-8') as log_file:
                log_file.write(f">>> SCREENSHOT CAPTURED at {current_time:.1f}s (time-based interval)\n")
            success = _save_screenshot(frame, current_time, screenshot_count, screenshots_dir)
            if success:
                screenshot_count += 1
        else:
            with open(log_path, 'a', encoding='utf-8') as log_file:
                log_file.write(f"Warning: Could not read frame at {current_time:.1f}s\n")
            if current_time + interval >= duration:
                break
        
        # Move to next time interval
        current_time += interval
        
        # Update progress indicator every 0.2 seconds
        current_real_time = time.time()
        if current_real_time - last_progress_update >= 0.2:
            progress_percent = (current_time / duration) * 100
            elapsed_time = current_real_time - process_start_time
            if progress_percent > 0:
                estimated_total_time = elapsed_time * (100 / progress_percent)
                remaining_time = estimated_total_time - elapsed_time
                remaining_str = f"{int(remaining_time//60):02d}:{int(remaining_time%60):02d}"
            else:
                remaining_str = "--:--"
            
            bar_length = 40
            filled_length = int(bar_length * progress_percent / 100)
            bar = '█' * filled_length + '-' * (bar_length - filled_length)
            print(f"\rProgress: [{bar}] {progress_percent:.1f}% ({current_time:.1f}s/{duration:.1f}s) - {screenshot_count} screenshots - ETA: {remaining_str}", end='', flush=True)
            last_progress_update = current_real_time
    
    # Final log entry
    with open(log_path, 'a', encoding='utf-8') as log_file:
        log_file.write(f"\n" + "=" * 50 + "\n")
        log_file.write(f"Analysis completed at {duration:.1f}s\n")
        log_file.write(f"Total screenshots captured: {screenshot_count}\n")
    
    # Remove duplicate screenshots
    removed_count = remove_duplicate_screenshots(screenshots_dir, top_ratio=0.2)
    final_count = screenshot_count - removed_count
    
    print(f"\n\nCompleted! Extracted {screenshot_count} screenshots, removed {removed_count} duplicates.")
    print(f"Final count: {final_count} unique screenshots.")
    print(f"Screenshots saved in: {screenshots_dir}")
    print(f"Log saved to: {log_path}")
    return True

def _extract_change_based(cap, fps, duration, start_time, change_threshold, screenshots_dir, screenshot_count, log_path, sanitized_name):
    """Extract screenshots based on content changes"""
    import time
    
    # Start from the beginning or specified start time
    cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
    
    previous_frame = None
    frame_count = 0
    min_interval_frames = int(fps * 2)  # Minimum 2 seconds between screenshots
    frames_since_last_screenshot = 0
    check_interval_frames = max(1, int(fps * 0.2))  # Check every 0.2 seconds
    
    print(f"Analyzing video for content changes (threshold: {change_threshold*100}%)...")
    print(f"Checking frames every {0.2}s ({check_interval_frames} frames)")
    
    # Initialize log file
    with open(log_path, 'w', encoding='utf-8') as log_file:
        log_file.write(f"Video Analysis Log - {sanitized_name}\n")
        log_file.write(f"Video Duration: {duration:.2f} seconds\n")
        log_file.write(f"Change Threshold: {change_threshold*100}%\n")
        log_file.write(f"Start Time: {start_time}s\n")
        log_file.write("=" * 50 + "\n\n")
    
    # Progress tracking
    process_start_time = time.time()
    last_progress_update = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        frames_since_last_screenshot += 1
        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        
        # Update progress indicator every 0.2 seconds (real time)
        current_real_time = time.time()
        if current_real_time - last_progress_update >= 0.2:
            progress_percent = (current_time / duration) * 100
            elapsed_time = current_real_time - process_start_time
            if progress_percent > 0:
                estimated_total_time = elapsed_time * (100 / progress_percent)
                remaining_time = estimated_total_time - elapsed_time
                remaining_str = f"{int(remaining_time//60):02d}:{int(remaining_time%60):02d}"
            else:
                remaining_str = "--:--"
            
            bar_length = 40
            filled_length = int(bar_length * progress_percent / 100)
            bar = '█' * filled_length + '-' * (bar_length - filled_length)
            print(f"\rProgress: [{bar}] {progress_percent:.1f}% ({current_time:.1f}s/{duration:.1f}s) - {screenshot_count} screenshots - ETA: {remaining_str}", end='', flush=True)
            last_progress_update = current_real_time
        
        # Check for changes every 0.2 seconds
        if frame_count % check_interval_frames == 0:
            if previous_frame is not None:
                change_percentage = _calculate_frame_change_percentage(previous_frame, frame, 0.2)
                
                # Log to file
                with open(log_path, 'a', encoding='utf-8') as log_file:
                    log_file.write(f"Time {current_time:.1f}s: {change_percentage:.1f}% different\n")
                
                # Check if change is over 70% - abort if so
                if change_percentage >= 70.0:
                    with open(log_path, 'a', encoding='utf-8') as log_file:
                        log_file.write(f"\n>>> MAJOR SCENE CHANGE DETECTED at {current_time:.1f}s ({change_percentage:.1f}% >= 70%)\n")
                        log_file.write(f">>> ABORTING CAPTURE PROCESS\n")
                    print(f"\n\n>>> MAJOR SCENE CHANGE DETECTED at {current_time:.1f}s ({change_percentage:.1f}% >= 70%)")
                    print(">>> ABORTING CAPTURE PROCESS - generating PDF with current screenshots")
                    break
                
                # Check if change exceeds threshold and minimum interval has passed
                if change_percentage >= (change_threshold * 100) and frames_since_last_screenshot >= min_interval_frames:
                    with open(log_path, 'a', encoding='utf-8') as log_file:
                        log_file.write(f"\n>>> SCREENSHOT CAPTURED at {current_time:.1f}s ({change_percentage:.1f}% >= {change_threshold*100}%)\n")
                    success = _save_screenshot(frame, current_time, screenshot_count, screenshots_dir)
                    if success:
                        screenshot_count += 1
                        frames_since_last_screenshot = 0
            else:
                # Save first frame after start time
                with open(log_path, 'a', encoding='utf-8') as log_file:
                    log_file.write(f"\n>>> INITIAL SCREENSHOT CAPTURED at {current_time:.1f}s\n")
                success = _save_screenshot(frame, current_time, screenshot_count, screenshots_dir)
                if success:
                    screenshot_count += 1
                    frames_since_last_screenshot = 0
        
        # Update previous frame every 0.2 seconds for comparison
        if frame_count % check_interval_frames == 0:
            previous_frame = frame.copy()
    
    # Final progress update
    print(f"\rProgress: [{'█' * 40}] 100.0% ({duration:.1f}s/{duration:.1f}s) - {screenshot_count} screenshots")
    
    # Final log entry
    with open(log_path, 'a', encoding='utf-8') as log_file:
        log_file.write(f"\n" + "=" * 50 + "\n")
        log_file.write(f"Analysis completed at {duration:.1f}s\n")
        log_file.write(f"Total screenshots captured: {screenshot_count}\n")
    
    # Remove duplicate screenshots
    removed_count = remove_duplicate_screenshots(screenshots_dir, top_ratio=0.2)
    final_count = screenshot_count - removed_count
    
    print(f"\nCompleted! Extracted {screenshot_count} screenshots using change detection, removed {removed_count} duplicates.")
    print(f"Final count: {final_count} unique screenshots.")
    print(f"Screenshots saved in: {screenshots_dir}")
    print(f"Log saved to: {log_path}")
    return True

def _calculate_frame_change_percentage(frame1, frame2, top_ratio=0.2):
    """
    Calculate the percentage of pixels that changed in the top portion of frames.
    
    Args:
        frame1: Previous frame (numpy array)
        frame2: Current frame (numpy array)
        top_ratio: Ratio of top portion to analyze (default: 0.2 = 20%)
    
    Returns:
        float: Percentage of changed pixels (0-100)
    """
    if frame1 is None or frame2 is None:
        return 100.0
    
    # Get dimensions
    height, width = frame1.shape[:2]
    top_height = int(height * top_ratio)
    
    # Extract top portions
    top1 = frame1[0:top_height, 0:width]
    top2 = frame2[0:top_height, 0:width]
    
    # Convert to grayscale for comparison
    gray1 = cv2.cvtColor(top1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(top2, cv2.COLOR_BGR2GRAY)
    
    # Calculate absolute difference
    diff = cv2.absdiff(gray1, gray2)
    
    # Threshold the difference (pixels with change > 30 intensity levels)
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    
    # Calculate percentage of changed pixels
    total_pixels = top_height * width
    changed_pixels = cv2.countNonZero(thresh)
    change_percentage = (changed_pixels / total_pixels) * 100
    
    return change_percentage

def remove_duplicate_screenshots(screenshots_dir, similarity_threshold=0.04, top_ratio=0.2):
    """
    Remove duplicate screenshots by comparing their similarity.
    Only compares the top portion of the images as specified by top_ratio.
    
    Args:
        screenshots_dir (str): Directory containing screenshots
        similarity_threshold (float): Threshold for considering images as duplicates (0.0-1.0)
        top_ratio (float): Ratio of top portion to analyze (default: 0.2 = 20%)
        
    Returns:
        int: Number of duplicate screenshots removed
    """
    import hashlib
    from pathlib import Path
    
    # Get all screenshot files
    screenshot_files = [f for f in Path(screenshots_dir).glob("*.jpg") if "screenshot_" in f.name]
    screenshot_files = sorted(screenshot_files, key=lambda x: int(x.name.split('_')[1]))  # Sort by sequence number
    
    if len(screenshot_files) <= 1:
        return 0
    
    removed_count = 0
    kept_files = [screenshot_files[0]]  # Always keep the first screenshot
    
    print(f"Checking {len(screenshot_files)} screenshots for duplicates...")
    
    for i in range(1, len(screenshot_files)):
        current_file = screenshot_files[i]
        is_duplicate = False
        
        # Compare with all previously kept files
        for kept_file in kept_files:
            if are_images_similar(str(kept_file), str(current_file), similarity_threshold, top_ratio):
                print(f"  Removing duplicate: {current_file.name} (similar to {kept_file.name})")
                os.remove(current_file)
                removed_count += 1
                is_duplicate = True
                break
        
        if not is_duplicate:
            kept_files.append(current_file)
    
    print(f"Removed {removed_count} duplicate screenshots")
    return removed_count


def are_images_similar(image1_path, image2_path, threshold=0.04, top_ratio=0.2):
    """
    Compare two images and return True if they are similar (below the change threshold).
    Uses the same logic as detect_frame_change but for image files.
    
    Args:
        image1_path (str): Path to first image
        image2_path (str): Path to second image
        threshold (float): Threshold for considering change significant (default: 0.04 = 4%)
        top_ratio (float): Ratio of top portion to analyze (default: 0.2 = 20%)
        
    Returns:
        bool: True if images are similar (change percentage is below threshold)
    """
    try:
        img1 = cv2.imread(image1_path)
        img2 = cv2.imread(image2_path)
        
        if img1 is None or img2 is None:
            return False
        
        # Get dimensions
        height1, width1 = img1.shape[:2]
        height2, width2 = img2.shape[:2]
        
        # Calculate top height for both images
        top_height1 = int(height1 * top_ratio)
        top_height2 = int(height2 * top_ratio)
        
        # Extract top portions
        top1 = img1[0:top_height1, 0:width1]
        top2 = img2[0:top_height2, 0:width2]
        
        # Resize top portions to same dimensions if needed
        if top1.shape[:2] != top2.shape[:2]:
            # Use the smaller dimensions to resize both
            min_height = min(top_height1, top_height2)
            min_width = min(width1, width2)
            
            top1 = cv2.resize(top1, (min_width, min_height))
            top2 = cv2.resize(top2, (min_width, min_height))
        
        # Convert to grayscale for comparison
        gray1 = cv2.cvtColor(top1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(top2, cv2.COLOR_BGR2GRAY)
        
        # Calculate absolute difference
        diff = cv2.absdiff(gray1, gray2)
        
        # Threshold the difference (pixels with change > 30 intensity levels)
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        
        # Calculate percentage of changed pixels
        total_pixels = top1.shape[0] * top1.shape[1]  # height * width
        changed_pixels = cv2.countNonZero(thresh)
        change_percentage = changed_pixels / total_pixels
        
        # Return True if the change percentage is below the threshold (meaning they're similar)
        return change_percentage < threshold
    except Exception as e:
        # If comparison fails for any reason, assume they're not duplicates
        return False


def _save_screenshot(frame, current_time, screenshot_count, screenshots_dir):
    """Save a screenshot with robust error handling"""
    # Check if frame is valid
    if frame is None or frame.size == 0:
        return False
        
    # Create filename with timestamp
    timestamp_str = f"{int(current_time//60):02d}m{int(current_time%60):02d}s"
    filename = f"screenshot_{screenshot_count+1:03d}_{timestamp_str}.jpg"
    filepath = os.path.join(screenshots_dir, filename)
    
    # Save the frame
    success = cv2.imwrite(filepath, frame)
    
    # If standard save failed, try alternative methods
    if not success or not os.path.exists(filepath):
        # Method 1: Try with encoded filename
        try:
            encoded_filepath = filepath.encode('utf-8').decode('utf-8')
            success = cv2.imwrite(encoded_filepath, frame)
            if success and os.path.exists(encoded_filepath):
                filepath = encoded_filepath
            else:
                raise Exception("UTF-8 encoding failed")
        except Exception as e:
            # Method 2: Use cv2.imencode and write manually
            try:
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    with open(filepath, 'wb') as f:
                        f.write(buffer)
                else:
                    raise Exception("cv2.imencode failed")
            except Exception as e:
                # Method 3: Use temporary filename and rename
                try:
                    temp_filename = f"temp_screenshot_{screenshot_count+1}.jpg"
                    temp_filepath = os.path.join(screenshots_dir, temp_filename)
                    
                    success = cv2.imwrite(temp_filepath, frame)
                    if success and os.path.exists(temp_filepath):
                        try:
                            os.rename(temp_filepath, filepath)
                        except:
                            filepath = temp_filepath
                            filename = temp_filename
                    else:
                        return False
                except Exception as e:
                    return False
    
    # Check if file was actually created
    return os.path.exists(filepath)

def main():
    parser = argparse.ArgumentParser(description="Extract screenshots from video files at regular intervals or based on content changes")
    parser.add_argument("video_path", help="Path to the video file")
    parser.add_argument("--start", type=float, default=2, help="Starting time in seconds (default: 2)")
    parser.add_argument("--interval", type=float, default=12, help="Interval between screenshots in seconds for time-based method (default: 12)")
    parser.add_argument("--method", choices=['time', 'change'], default='change', help="Screenshot method: 'time' for fixed intervals, 'change' for content change detection (default: change)")
    parser.add_argument("--change-threshold", type=float, default=0.04, help="Threshold for change detection method (0.0-1.0, default: 0.04 = 4%)")
    parser.add_argument("--test", action="store_true", help="Test mode: only check video properties without extracting screenshots")
    parser.add_argument("--create-pdf", action="store_true", help="Create PDF after extracting screenshots")
    parser.add_argument("--crop-ratio", type=float, default=0.32, help="Portion of height to keep from top for PDF (default: 0.32)")
    parser.add_argument("--strips-per-page", type=int, default=6, help="Maximum strips per A4 page (default: 6)")
    
    args = parser.parse_args()
    
    # Check if file exists first
    if not os.path.exists(args.video_path):
        print(f"Error: File '{args.video_path}' does not exist.")
        print("Please check the file path and try again.")
        sys.exit(1)
    
    # Check file size
    file_size = os.path.getsize(args.video_path)
    print(f"File size: {file_size:,} bytes ({file_size / (1024*1024):.2f} MB)")
    
    # Validate video file extension
    if not args.video_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
        print("Warning: File doesn't appear to be a video file. Supported formats: .mp4, .avi, .mov, .mkv, .webm")
    
    if args.test:
        # Test mode: just check if we can open the video and read its properties
        print("=== TEST MODE ===")
        
        cap = cv2.VideoCapture(args.video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file '{args.video_path}'")
            sys.exit(1)
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video properties:")
        print(f"  Duration: {duration:.2f} seconds")
        print(f"  FPS: {fps:.2f}")
        print(f"  Total frames: {total_frames}")
        print(f"  Resolution: {width}x{height}")
        
        # Try to read the first frame
        ret, frame = cap.read()
        if ret:
            print(f"  ✓ Successfully read first frame")
        else:
            print(f"  ✗ Could not read first frame")
        
        cap.release()
        return
    
    # Extract screenshots
    result = extract_screenshots(
        args.video_path, 
        args.start, 
        args.interval, 
        args.method, 
        args.change_threshold
    )
    
    if isinstance(result, tuple):
        success, screenshots_dir = result
    else:
        success = result
        # Fallback to old structure if needed
        video_name = Path(args.video_path).stem
        sanitized_name = sanitize_filename(video_name)
        screenshots_dir = os.path.join(os.getcwd(), f"{sanitized_name}_screenshots")
    
    if not success:
        sys.exit(1)
    
    # Create PDF if requested
    if args.create_pdf:
        print("\n=== CREATING PDF ===")
        video_name = Path(args.video_path).stem
        sanitized_name = sanitize_filename(video_name)
        main_folder = os.path.join(os.getcwd(), sanitized_name)
        screenshots_dir = os.path.join(main_folder, "screenshots")
        
        if os.path.exists(screenshots_dir):
            # Import create_pdf function directly instead of subprocess
            try:
                # Import the create_pdf module functions
                import importlib.util
                spec = importlib.util.spec_from_file_location("create_pdf", "create_pdf.py")
                if spec is not None and spec.loader is not None:
                    create_pdf_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(create_pdf_module)
                    
                    pdf_filename = f"{sanitized_name}_score.pdf"
                    pdf_path = os.path.join(main_folder, pdf_filename)
                    
                    print(f"Creating PDF: {pdf_path}")
                    success = create_pdf_module.create_pdf_from_screenshots(
                        screenshots_dir, 
                        pdf_path, 
                        args.crop_ratio, 
                        args.strips_per_page,
                        sanitized_name  # Pass song title
                    )
                    
                    if success:
                        print(f"✓ PDF created successfully: {pdf_path}")
                    else:
                        print(f"✗ PDF creation failed")
                else:
                    raise Exception("Could not load create_pdf module")
                    
            except Exception as e:
                print(f"✗ Error creating PDF: {e}")
                print("Trying alternative subprocess method...")
                
                # Fallback to subprocess with proper encoding
                try:
                    import subprocess
                    pdf_filename = f"{sanitized_name}_stacked.pdf"
                    
                    # Run the PDF creation script with proper encoding
                    cmd = [
                        sys.executable, "create_pdf.py", screenshots_dir, 
                        "--output", pdf_filename,
                        "--crop-ratio", str(args.crop_ratio),
                        "--strips-per-page", str(args.strips_per_page)
                    ]
                    
                    result = subprocess.run(
                        cmd, 
                        capture_output=True, 
                        text=True, 
                        cwd=os.getcwd(),
                        encoding='utf-8',
                        errors='replace'
                    )
                    
                    if result.returncode == 0:
                        print(f"✓ PDF created successfully: {pdf_filename}")
                    else:
                        print(f"✗ PDF creation failed:")
                        print(result.stderr)
                        
                except Exception as e2:
                    print(f"✗ Both PDF creation methods failed: {e2}")
        else:
            print(f"✗ Screenshots directory not found: {screenshots_dir}")

if __name__ == "__main__":
    main()