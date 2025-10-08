import cv2
import numpy as np
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
    #   ├── debug_masks/
    #   └── [video_name]_score.pdf
    main_folder = os.path.join(os.getcwd(), sanitized_name)
    screenshots_dir = os.path.join(main_folder, "screenshots")
    debug_masks_dir = os.path.join(main_folder, "debug_masks")
    
    # Delete existing folder if it exists
    if os.path.exists(main_folder):
        import shutil
        print(f"Existing folder found: {main_folder}")
        print("Deleting existing folder and contents...")
        shutil.rmtree(main_folder)
        print("✓ Existing folder deleted")
    
    # Create new folder structure
    os.makedirs(screenshots_dir, exist_ok=True)
    os.makedirs(debug_masks_dir, exist_ok=True)
    print(f"Created main folder: {main_folder}")
    print(f"Screenshots will be saved to: {screenshots_dir}")
    print(f"Debug masks will be saved to: {debug_masks_dir}")
    
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
        success = _extract_time_based(cap, fps, duration, start_time, interval, screenshots_dir, screenshot_count, log_path, sanitized_name, debug_masks_dir)
    else:
        print(f"Using change-based method: detecting {change_threshold*100}% change in top 20% of frame")
        print(f"Log file will be saved to: {log_path}")
        success = _extract_change_based(cap, fps, duration, start_time, change_threshold, screenshots_dir, screenshot_count, log_path, sanitized_name, debug_masks_dir)
    
    cap.release()
    return success, screenshots_dir

def _extract_time_based(cap, fps, duration, start_time, interval, screenshots_dir, screenshot_count, log_path, sanitized_name, debug_masks_dir):
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
            success = _save_screenshot_with_progress_bar_merge(cap, current_time, screenshot_count, screenshots_dir, log_path, debug_masks_dir)
            if success:
                screenshot_count += 1
        else:
            with open(log_path, 'a', encoding='utf-8') as log_file:
                log_file.write(f"Warning: Could not read frame at {current_time:.1f}s\n")
            if current_time + interval >= duration:
                break
        
        # Move to next time interval
        current_time += interval
        
        # Update progress indicator every 0.1 seconds
        current_real_time = time.time()
        if current_real_time - last_progress_update >= 0.1:
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
    remaining_count = remove_duplicate_screenshots(screenshots_dir, log_path)
    
    print(f"\n\nCompleted! Extracted {screenshot_count} screenshots.")
    print(f"After duplicate removal: {remaining_count} unique screenshots")
    print(f"Screenshots saved in: {screenshots_dir}")
    print(f"Log saved to: {log_path}")
    return True

def _extract_change_based(cap, fps, duration, start_time, change_threshold, screenshots_dir, screenshot_count, log_path, sanitized_name, debug_masks_dir):
    """Extract screenshots based on content changes"""
    import time
    
    # Start from the beginning or specified start time
    cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
    
    previous_frame = None
    frame_count = 0
    min_interval_frames = int(fps * 2)  # Minimum 2 seconds between screenshots
    frames_since_last_screenshot = 0
    check_interval_frames = max(1, int(fps * 1.0))  # Check every 1.0 second
    
    print(f"Analyzing video for content changes (threshold: {change_threshold*100}%)...")
    print(f"Checking frames every {1.0}s ({check_interval_frames} frames)")
    
    # Initialize log file
    with open(log_path, 'w', encoding='utf-8') as log_file:
        log_file.write(f"Video Analysis Log - {sanitized_name}\n")
        log_file.write(f"Video Duration: {duration:.2f} seconds\n")
        log_file.write(f"Change Threshold: {change_threshold*100}%\n")
        log_file.write(f"Start Time: {start_time}s\n")
        log_file.write(f"Check Interval: 1.0s\n")
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
        
        # Update progress indicator every 0.1 seconds (real time)
        current_real_time = time.time()
        if current_real_time - last_progress_update >= 0.1:
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
        
        # Check for changes every 1.0 second
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
                    success = _save_screenshot_with_progress_bar_merge(cap, current_time, screenshot_count, screenshots_dir, log_path, debug_masks_dir)
                    if success:
                        screenshot_count += 1
                        frames_since_last_screenshot = 0
            else:
                # Save first frame after start time
                with open(log_path, 'a', encoding='utf-8') as log_file:
                    log_file.write(f"\n>>> INITIAL SCREENSHOT CAPTURED at {current_time:.1f}s\n")
                success = _save_screenshot_with_progress_bar_merge(cap, current_time, screenshot_count, screenshots_dir, log_path, debug_masks_dir)
                if success:
                    screenshot_count += 1
                    frames_since_last_screenshot = 0
        
        # Update previous frame every 1.0 second for comparison
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
    remaining_count = remove_duplicate_screenshots(screenshots_dir, log_path)
    
    print(f"\nCompleted! Extracted {screenshot_count} screenshots using change detection.")
    print(f"After duplicate removal: {remaining_count} unique screenshots")
    print(f"Screenshots saved in: {screenshots_dir}")
    print(f"Log saved to: {log_path}")
    return True

def compare_screenshots_similarity(image1_path, image2_path, top_ratio=0.2, threshold=0.04):
    """
    Compare two screenshots to determine if they are duplicates.
    
    Args:
        image1_path: Path to first image
        image2_path: Path to second image
        top_ratio: Ratio of top portion to compare (default: 0.2 = 20%)
        threshold: Threshold for considering images different (default: 0.04 = 4%)
    
    Returns:
        bool: True if images are similar (duplicates), False if different
    """
    try:
        # Load images
        img1 = cv2.imread(image1_path)
        img2 = cv2.imread(image2_path)
        
        if img1 is None or img2 is None:
            return False
        
        # Resize images to same dimensions if needed
        if img1.shape != img2.shape:
            height = min(img1.shape[0], img2.shape[0])
            width = min(img1.shape[1], img2.shape[1])
            img1 = cv2.resize(img1, (width, height))
            img2 = cv2.resize(img2, (width, height))
        
        # Get top portion only
        height, width = img1.shape[:2]
        top_height = int(height * top_ratio)
        
        top1 = img1[0:top_height, 0:width]
        top2 = img2[0:top_height, 0:width]
        
        # Convert to grayscale
        gray1 = cv2.cvtColor(top1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(top2, cv2.COLOR_BGR2GRAY)
        
        # Calculate absolute difference
        diff = cv2.absdiff(gray1, gray2)
        
        # Threshold the difference
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        
        # Calculate percentage of different pixels
        total_pixels = top_height * width
        different_pixels = cv2.countNonZero(thresh)
        difference_percentage = (different_pixels / total_pixels)
        
        # Return True if images are similar (difference < threshold)
        return difference_percentage < threshold
        
    except Exception as e:
        print(f"Error comparing {image1_path} and {image2_path}: {e}")
        return False

def remove_duplicate_screenshots(screenshots_dir, log_path):
    """
    Remove duplicate screenshots, keeping the one taken earlier in time.
    
    Args:
        screenshots_dir: Directory containing screenshots
        log_path: Path to log file
    
    Returns:
        int: Number of remaining screenshots after duplicate removal
    """
    import glob
    import os
    import re
    
    # Get all screenshot files
    screenshot_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        screenshot_files.extend(glob.glob(os.path.join(screenshots_dir, ext)))
    
    if len(screenshot_files) < 2:
        with open(log_path, 'a', encoding='utf-8') as log_file:
            log_file.write(f"\nDuplicate Detection: {len(screenshot_files)} screenshots found, no duplicates to check\n")
        return len(screenshot_files)
    
    # Sort files by timestamp (extract from filename)
    def extract_timestamp(filename):
        # Extract timestamp from filename like "screenshot_001_05m23s.jpg"
        basename = os.path.basename(filename)
        match = re.search(r'(\d+)m(\d+)s', basename)
        if match:
            minutes = int(match.group(1))
            seconds = int(match.group(2))
            return minutes * 60 + seconds
        return 0
    
    screenshot_files.sort(key=extract_timestamp)
    
    with open(log_path, 'a', encoding='utf-8') as log_file:
        log_file.write(f"\nDuplicate Detection: Checking {len(screenshot_files)} screenshots for duplicates\n")
        log_file.write("Comparing top 20% of images with 4% threshold\n")
    
    duplicates_removed = 0
    removed_files = []
    
    # Compare each image with all others
    for i in range(len(screenshot_files)):
        if screenshot_files[i] in removed_files:
            continue
            
        for j in range(i + 1, len(screenshot_files)):
            if screenshot_files[j] in removed_files:
                continue
                
            # Compare the two images
            are_similar = compare_screenshots_similarity(screenshot_files[i], screenshot_files[j])
            
            if are_similar:
                # Remove the later screenshot (higher index = later timestamp)
                file_to_remove = screenshot_files[j]
                try:
                    os.remove(file_to_remove)
                    removed_files.append(file_to_remove)
                    duplicates_removed += 1
                    
                    with open(log_path, 'a', encoding='utf-8') as log_file:
                        log_file.write(f"Duplicate removed: {os.path.basename(file_to_remove)} (duplicate of {os.path.basename(screenshot_files[i])})\n")
                        
                except Exception as e:
                    with open(log_path, 'a', encoding='utf-8') as log_file:
                        log_file.write(f"Error removing duplicate {file_to_remove}: {e}\n")
    
    remaining_count = len(screenshot_files) - duplicates_removed
    
    with open(log_path, 'a', encoding='utf-8') as log_file:
        log_file.write(f"Duplicate removal completed: {duplicates_removed} duplicates removed, {remaining_count} unique screenshots remaining\n")
    
    return remaining_count

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

def detect_progress_bar(frame, top_ratio=0.32):
    """
    Detect vertical progress bar in the top portion of frame.
    Progress bar is typically a thin vertical colored line that stands out from the music notation.
    
    Args:
        frame: OpenCV frame (numpy array)
        top_ratio: Ratio of top portion to analyze (default: 0.32 = 32%)
    
    Returns:
        tuple: (x_start, x_end, detected) where x_start and x_end are the horizontal 
               boundaries of the progress bar, and detected is True if found
    """
    if frame is None:
        return None, None, False
    
    height, width = frame.shape[:2]
    top_height = int(height * top_ratio)
    top_region = frame[0:top_height, 0:width]
    
    # Convert to different color spaces for better analysis
    hsv = cv2.cvtColor(top_region, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(top_region, cv2.COLOR_BGR2GRAY)
    
    # Method 1: Look for vertical edges (thin lines)
    # Apply vertical edge detection
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_x = np.abs(sobel_x)
    sobel_x = (sobel_x / sobel_x.max() * 255).astype(np.uint8)
    
    # Threshold to find strong vertical edges
    _, edge_mask = cv2.threshold(sobel_x, 100, 255, cv2.THRESH_BINARY)
    
    # Method 2: Look for distinct colors that stand out
    # Create mask for pixels that are significantly different from background
    # Analyze color variance in small vertical strips
    progress_bar_candidates = []
    
    # Scan in vertical strips across the width
    strip_width = 3  # Analyze 3-pixel wide strips
    for x in range(0, width - strip_width, 2):  # Step by 2 pixels
        strip = top_region[:, x:x+strip_width]
        strip_hsv = hsv[:, x:x+strip_width]
        strip_edges = edge_mask[:, x:x+strip_width]
        
        # Check if this strip has strong vertical edges
        edge_density = np.sum(strip_edges > 0) / (top_height * strip_width)
        
        # Check color uniformity in the strip (progress bars are usually uniform color)
        if strip.size > 0:
            # Calculate color variance
            mean_color = np.mean(strip, axis=(0, 1))
            color_variance = np.var(strip.reshape(-1, 3), axis=0)
            total_variance = np.sum(color_variance)
            
            # Check if the strip has uniform color and strong edges
            if edge_density > 0.3 and total_variance < 1000:  # Uniform color, strong edges
                # Check if it's not pure black or white
                if not (np.all(mean_color < 10) or np.all(mean_color > 300)):
                    # Calculate height of continuous colored region
                    strip_gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
                    
                    # Find the longest continuous vertical line
                    max_continuous_height = 0
                    current_height = 0
                    
                    for y in range(top_height):
                        if np.any(strip_edges[y, :] > 0):  # Has edge
                            current_height += 1
                        else:
                            max_continuous_height = max(max_continuous_height, current_height)
                            current_height = 0
                    max_continuous_height = max(max_continuous_height, current_height)
                    
                    # If we found a significant vertical line
                    if max_continuous_height > top_height * 0.3:  # At least 30% of region height
                        progress_bar_candidates.append({
                            'x_start': x,
                            'x_end': x + strip_width,
                            'width': strip_width,
                            'height': max_continuous_height,
                            'edge_density': edge_density,
                            'color_variance': total_variance,
                            'mean_color': mean_color,
                            'score': edge_density * max_continuous_height / total_variance if total_variance > 0 else 0
                        })
    
    # Method 3: Look for thin vertical contours with distinct colors
    # Create a more sophisticated color mask
    lab = cv2.cvtColor(top_region, cv2.COLOR_BGR2LAB)
    
    # Find regions that are colorimetrically distinct
    # Calculate local color contrast
    contrast_mask = np.zeros(gray.shape, dtype=np.uint8)
    
    kernel_size = 5
    for y in range(kernel_size, top_height - kernel_size):
        for x in range(kernel_size, width - kernel_size):
            # Get local neighborhood
            neighborhood = lab[y-kernel_size:y+kernel_size+1, x-kernel_size:x+kernel_size+1]
            center_pixel = lab[y, x]
            
            # Calculate color distance from center to neighbors
            distances = np.sqrt(np.sum((neighborhood - center_pixel) ** 2, axis=2))
            max_distance = np.max(distances)
            
            # If this pixel is very different from its surroundings
            if max_distance > 30:  # Threshold for color distinctness
                contrast_mask[y, x] = 255
    
    # Find contours in contrast mask
    contours, _ = cv2.findContours(contrast_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        # Look for thin vertical shapes
        if (h > w * 2 and  # Height > 2x width (more vertical)
            w >= 1 and w <= 20 and  # Very thin (1-20 pixels wide)
            h >= top_height * 0.2):  # Significant height
            
            area_ratio = cv2.contourArea(contour) / (w * h) if w * h > 0 else 0
            if area_ratio > 0.3:  # Reasonably filled
                progress_bar_candidates.append({
                    'x_start': x,
                    'x_end': x + w,
                    'width': w,
                    'height': h,
                    'edge_density': 1.0,  # From contour detection
                    'color_variance': 0,   # Assume uniform
                    'mean_color': np.mean(top_region[y:y+h, x:x+w], axis=(0,1)),
                    'score': h * (1 / w) * area_ratio  # Prefer tall, thin, filled shapes
                })
    
    # Select the best candidate
    if progress_bar_candidates:
        # Sort by score (higher is better)
        progress_bar_candidates.sort(key=lambda x: x['score'], reverse=True)
        
        best_candidate = progress_bar_candidates[0]
        return best_candidate['x_start'], best_candidate['x_end'], True
    
    return None, None, False

def detect_progress_bar_with_debug(frame, current_time, screenshot_count, debug_masks_dir):
    """
    Detect vertical progress bar and save debug visualization images.
    
    Args:
        frame: OpenCV frame (numpy array)
        current_time: Current timestamp
        screenshot_count: Current screenshot number
        debug_masks_dir: Directory to save debug images
    
    Returns:
        tuple: (x_start, x_end, detected) same as detect_progress_bar
    """
    if frame is None:
        return None, None, False
    
    top_ratio = 0.32
    height, width = frame.shape[:2]
    top_height = int(height * top_ratio)
    top_region = frame[0:top_height, 0:width]
    
    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(top_region, cv2.COLOR_BGR2HSV)
    
    # Create mask for non-black and non-white pixels
    black_threshold = 190
    white_value_threshold = 250 
    white_saturation_threshold = 40  
    
    # Create masks
    not_black_mask = hsv[:, :, 2] > black_threshold
    not_white_mask = ~((hsv[:, :, 2] > white_value_threshold) & (hsv[:, :, 1] < white_saturation_threshold))
    colored_mask = not_black_mask & not_white_mask
    colored_mask = colored_mask.astype(np.uint8) * 255
    
    # Apply morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    colored_mask_cleaned = cv2.morphologyEx(colored_mask, cv2.MORPH_CLOSE, kernel)
    colored_mask_cleaned = cv2.morphologyEx(colored_mask_cleaned, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(colored_mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create debug visualization
    debug_image = top_region.copy()
    mask_overlay = cv2.cvtColor(colored_mask_cleaned, cv2.COLOR_GRAY2BGR)
    
    # Look for progress bar candidates
    progress_bar_candidates = []
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        if (h > w * 1.5 and w >= 8 and w <= 60 and h >= 30 and y < top_height * 0.9):
            contour_area = cv2.contourArea(contour)
            bbox_area = w * h
            area_ratio = contour_area / bbox_area if bbox_area > 0 else 0
            
            if area_ratio > 0.3:
                roi_mask = colored_mask_cleaned[y:y+h, x:x+w]
                colored_pixels = np.sum(roi_mask > 0)
                total_pixels = w * h
                color_density = colored_pixels / total_pixels
                
                if color_density > 0.5:
                    progress_bar_candidates.append((x, x + w, w, h, area_ratio, color_density))
                    # Draw candidate rectangles in blue
                    cv2.rectangle(debug_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(debug_image, f"{color_density:.2f}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    # Determine best candidate
    detected = False
    x_start, x_end = None, None
    
    if progress_bar_candidates:
        progress_bar_candidates.sort(key=lambda x: (-x[5], -x[2] * x[3]))
        best_candidate = progress_bar_candidates[0]
        x_start, x_end = best_candidate[0], best_candidate[1]
        detected = True
        
        # Draw final selection in green
        y_pos = 0  # We don't store y in the return, but we can approximate
        for candidate in progress_bar_candidates:
            if candidate[0] == x_start and candidate[1] == x_end:
                # Find the contour that matches this candidate
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    if x == x_start and x + w == x_end:
                        cv2.rectangle(debug_image, (x, y), (x + w, y + h), (0, 255, 0), 3)
                        cv2.putText(debug_image, "SELECTED", (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        break
                break
    
    # Save debug images
    timestamp_str = f"{int(current_time//60):02d}m{int(current_time%60):02d}s"
    
    # Save original top region
    original_filename = f"debug_{screenshot_count+1:03d}_{timestamp_str}_original.jpg"
    original_path = os.path.join(debug_masks_dir, original_filename)
    cv2.imwrite(original_path, top_region)
    
    # Save color mask
    mask_filename = f"debug_{screenshot_count+1:03d}_{timestamp_str}_mask.jpg"
    mask_path = os.path.join(debug_masks_dir, mask_filename)
    cv2.imwrite(mask_path, mask_overlay)
    
    # Save detection result
    result_filename = f"debug_{screenshot_count+1:03d}_{timestamp_str}_result.jpg"
    result_path = os.path.join(debug_masks_dir, result_filename)
    cv2.imwrite(result_path, debug_image)
    
    return x_start, x_end, detected

def merge_progress_bar_strip(frame_a, frame_b, progress_bar_x_start, progress_bar_x_end, strip_width_factor=1.5):
    """
    Extract progress bar strip from frame_b and merge it into frame_a.
    
    Args:
        frame_a: Base frame (numpy array)
        frame_b: Frame with updated progress bar (numpy array)
        progress_bar_x_start: X coordinate of progress bar start
        progress_bar_x_end: X coordinate of progress bar end
        strip_width_factor: Factor to make strip wider than progress bar
    
    Returns:
        numpy array: Merged frame
    """
    if frame_a is None or frame_b is None:
        return frame_a
    
    if progress_bar_x_start is None or progress_bar_x_end is None:
        return frame_a
    
    # Calculate strip boundaries (make it wider than the progress bar)
    progress_bar_width = progress_bar_x_end - progress_bar_x_start
    extra_width = int(progress_bar_width * (strip_width_factor - 1) / 2)
    
    strip_x_start = max(0, progress_bar_x_start - extra_width)
    strip_x_end = min(frame_a.shape[1], progress_bar_x_end + extra_width)
    
    # Get the top 32% of both frames
    height = frame_a.shape[0]
    top_height = int(height * 0.32)
    
    # Create a copy of frame_a
    merged_frame = frame_a.copy()
    
    # Extract the strip from frame_b and overlay it on frame_a
    strip_from_b = frame_b[0:top_height, strip_x_start:strip_x_end]
    merged_frame[0:top_height, strip_x_start:strip_x_end] = strip_from_b
    
    return merged_frame

def _save_screenshot_with_progress_bar_merge(cap, current_time, screenshot_count, screenshots_dir, log_path, debug_masks_dir):
    """
    Capture screenshot A, wait 1 second, capture screenshot B, detect progress bar,
    merge progress bar from B into A, and save the result.
    
    Args:
        cap: Video capture object
        current_time: Current timestamp
        screenshot_count: Current screenshot number
        screenshots_dir: Directory to save screenshots
        log_path: Path to log file
        debug_masks_dir: Directory to save debug mask images
    
    Returns:
        bool: True if successful, False otherwise
    """
    import time
    
    # Get frame A at current time
    frame_number_a = int(current_time * cap.get(cv2.CAP_PROP_FPS))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number_a)
    ret_a, frame_a = cap.read()
    
    if not ret_a or frame_a is None:
        with open(log_path, 'a', encoding='utf-8') as log_file:
            log_file.write(f"Error: Could not capture frame A at {current_time:.1f}s\n")
        return False
    
    # Detect progress bar in frame A and save debug images
    progress_x_start, progress_x_end, progress_detected = detect_progress_bar_with_debug(frame_a, current_time, screenshot_count, debug_masks_dir)
    
    with open(log_path, 'a', encoding='utf-8') as log_file:
        if progress_detected and progress_x_start is not None and progress_x_end is not None:
            log_file.write(f"Progress bar detected at {current_time:.1f}s: x={progress_x_start}-{progress_x_end} (width={progress_x_end-progress_x_start}px)\n")
        else:
            log_file.write(f"No progress bar detected at {current_time:.1f}s\n")
    
    # If progress bar detected, wait 1 second and capture frame B
    final_frame = frame_a
    if progress_detected:
        # Wait 1 second
        time.sleep(1)
        
        # Calculate frame B position (1 second later)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_number_b = int((current_time + 1) * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number_b)
        ret_b, frame_b = cap.read()
        
        if ret_b and frame_b is not None:
            # Merge progress bar strip from frame B into frame A
            final_frame = merge_progress_bar_strip(frame_a, frame_b, progress_x_start, progress_x_end)
            with open(log_path, 'a', encoding='utf-8') as log_file:
                log_file.write(f"Progress bar merged from frame at {current_time + 1:.1f}s\n")
        else:
            with open(log_path, 'a', encoding='utf-8') as log_file:
                log_file.write(f"Warning: Could not capture frame B for progress bar merge\n")
    
    # Save the final frame
    return _save_screenshot(final_frame, current_time, screenshot_count, screenshots_dir)

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
    parser.add_argument("--method", choices=['time', 'change'], default='time', help="Screenshot method: 'time' for fixed intervals, 'change' for content change detection (default: time)")
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