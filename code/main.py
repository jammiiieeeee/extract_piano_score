import os
import sys

# ===== CRITICAL FIX: SET ENV FLAGS BEFORE IMPORTING PADDLE =====
# These must be set before 'from paddleocr import ...' runs
os.environ['FLAGS_use_mkldnn'] = '0'
os.environ['FLAGS_enable_mkldnn'] = '0'
os.environ['FLAGS_use_ngraph'] = 'False'
os.environ['PADDLE_DISABLE_STATIC'] = 'True'
# Disable the PIR (Program Intermediate Representation) to Runtime conversion if possible
os.environ['FLAGS_enable_pir_in_executor'] = '0'
os.environ['FLAGS_enable_pir_api'] = '0' 

import cv2
import argparse
import re
import numpy as np
import json
import time
import shutil
from pathlib import Path
from PIL import Image

# Import PaddleOCR after setting flags
try:
    from paddleocr import PaddleOCR
except ImportError:
    print("Error: PaddleOCR not installed. Please run: pip install paddlepaddle paddleocr")
    sys.exit(1)

# Initialize PaddleOCR with multiple fallback attempts
ocr = None
print("Initializing PaddleOCR...")

# Try multiple initialization strategies
# Added 'enable_mkldnn=False' explicitly to constructors
initialization_attempts = [
    # Attempt 1: Most conservative (CPU only, no MKLDNN, no angle class)
    lambda: PaddleOCR(
        lang='en', 
        use_gpu=False, 
        show_log=False, 
        enable_mkldnn=False,
        use_mp=True, 
        total_process_num=2
    ),
    # Attempt 2: Standard CPU with MKLDNN disabled
    lambda: PaddleOCR(
        lang='en', 
        use_gpu=False, 
        enable_mkldnn=False
    ),
    # Attempt 3: Default
    lambda: PaddleOCR(lang='en', enable_mkldnn=False),
    # Attempt 4: Bare minimum
    lambda: PaddleOCR(enable_mkldnn=False)
]

for i, init_func in enumerate(initialization_attempts):
    try:
        print(f"  Attempt {i+1}: ", end="")
        ocr = init_func()
        
        # specific check to ensure the model loaded
        if ocr:
            print("SUCCESS")
            break
    except Exception as e:
        print(f"FAILED - {e}")
        if i == len(initialization_attempts) - 1:
            print("All PaddleOCR initialization attempts failed!")
            print("Try running: python -m pip install --upgrade paddlepaddle paddleocr")
            ocr = None

# ===== CONFIGURATION THRESHOLDS =====
# All configurable thresholds grouped for easy access and modification

class Config:
    """Configuration class containing all thresholds and parameters"""
    
    # ---- Video Analysis Thresholds ----
    CHANGE_DETECTION_THRESHOLD = 0.04      # Default change detection threshold (4%)
    MAJOR_SCENE_CHANGE_THRESHOLD = 40.0    # Major scene change detection (70%)
    FRAME_CHECK_INTERVAL = 0.2             # Check frames every N seconds
    TOP_ANALYSIS_RATIO = 0.2               # Analyze top 20% of frame for changes
    MIN_SCREENSHOT_INTERVAL = 3.0          # Minimum seconds between screenshots
    VIDEO_END_BUFFER = 10.0                # Don't capture A screenshots in last N seconds
    
    # ---- Duplicate Detection Thresholds ----
    DUPLICATE_TOP_RATIO = 0.27             # Analyze top 27% for duplicate detection
    PIXEL_SIMILARITY_THRESHOLD = 0.95      # Test 1: 95% pixel similarity
    ROW_SIMILARITY_THRESHOLD = 0.98        # Test 2: 98% threshold per row
    ROW_COVERAGE_THRESHOLD = 0.94          # Test 2: 94% of rows must pass
    
    # ---- A/B Merging Parameters ----
    A_CAPTURE_DELAY = 0                 # Seconds delay for A screenshot after change detection
    B_CAPTURE_DELAY = 3                  # Seconds delay for B screenshot capture
    B_OVERLAY_WIDTH_RATIO = 0.2           # Use left 20% of B screenshot for overlay
    
    # ---- Image Processing Thresholds ----
    PIXEL_INTENSITY_THRESHOLD = 30         # Intensity threshold for binary diff
    PROGRESS_UPDATE_INTERVAL = 0.2         # Progress update every 0.2 seconds
    
    # ---- PDF Generation Defaults ----
    DEFAULT_CROP_RATIO = 0.32              # Default PDF crop ratio (32% from top)
    DEFAULT_STRIPS_PER_PAGE = 7            # Default strips per PDF page
    
    # ---- OCR Configuration ----
    OCR_CROP_RATIO = DEFAULT_CROP_RATIO     # Use same ratio as PDF crop for OCR analysis (top portion)
    OCR_HORIZONTAL_RATIO = 0.30              # Horizontal portion to analyze (0.5 = left 50%)
    OCR_CONFIDENCE_THRESHOLD = 40           # Minimum confidence for OCR text extraction (0-100, higher = more strict)
    # Note: OCR analyzes top OCR_CROP_RATIO and left OCR_HORIZONTAL_RATIO of image, returns only leftmost number
    # Confidence threshold: 0-30 = lenient, 30-60 = balanced, 60-100 = strict

# =====================================

def extract_ocr_numbers(image_path):
    """
    Extract the leftmost number from the top-left portion of an image using OCR.
    Returns tuple: (number_list, bbox_info) where bbox_info contains the bounding box of the detected number
    """
    try:
        # Load image
        image = Image.open(image_path)
        width, height = image.size
        
        # Crop to top portion using OCR_CROP_RATIO and configurable horizontal ratio
        crop_height = int(height * Config.OCR_CROP_RATIO)
        crop_width = int(width * Config.OCR_HORIZONTAL_RATIO)
        cropped_image = image.crop((0, 0, crop_width, crop_height))
        
        # Convert PIL image to numpy array for PaddleOCR
        img_array = np.array(cropped_image)
        
        # Extract text using PaddleOCR
        results = None
        high_confidence_numbers = []
        
        if ocr is not None:
            try:
                # 
                results = ocr.ocr(img_array)
                
                # Debug: print raw OCR results to help troubleshoot
                # print(f"  Debug OCR raw results for {os.path.basename(image_path)}: {results}")

                if not results:
                    return []

                # Collect numbers with their x-coordinates and bounding boxes for spatial sorting
                numbers_with_positions = []

                # --- FORMAT 1: New Dictionary Format (PaddleX / v2.7+) ---
                # Structure: [{'rec_texts': ['123', 'Text'], 'rec_scores': [0.99, 0.98], 'rec_polys': [bbox_array], ...}]
                if isinstance(results, list) and len(results) > 0 and isinstance(results[0], dict):
                    data = results[0]
                    # Check if 'rec_texts' and 'rec_scores' exist
                    if 'rec_texts' in data and 'rec_scores' in data:
                        texts = data['rec_texts']
                        scores = data['rec_scores']
                        polys = data.get('rec_polys', [])  # Get polygon coordinates if available
                        
                        for i, text in enumerate(texts):
                            score = scores[i]
                            # Filter by confidence
                            if score * 100 >= Config.OCR_CONFIDENCE_THRESHOLD:
                                # Extract numbers
                                numbers_in_text = re.findall(r'\d+', text)
                                # Get bounding box if available
                                bbox = polys[i] if i < len(polys) else None
                                if bbox is not None:
                                    # Convert numpy array to list format
                                    bbox_list = bbox.tolist() if hasattr(bbox, 'tolist') else bbox
                                    min_x = min(point[0] for point in bbox_list)
                                    for num in numbers_in_text:
                                        numbers_with_positions.append((int(num), min_x, bbox_list))
                                else:
                                    # No bbox available, use index as position
                                    for num in numbers_in_text:
                                        numbers_with_positions.append((int(num), i, None))

                # --- FORMAT 2: Legacy List Format ---
                # Structure: [[[[x,y],..], ('Text', 0.99)], ...]
                elif isinstance(results, list) and len(results) > 0 and isinstance(results[0], list):
                    for line in results[0]:
                        if line and len(line) >= 2:
                            bbox = line[0]  # Bounding box coordinates
                            text = line[1][0]
                            score = line[1][1]
                            
                            if score * 100 >= Config.OCR_CONFIDENCE_THRESHOLD:
                                numbers_in_text = re.findall(r'\d+', text)
                                # Get leftmost x-coordinate from bounding box
                                if bbox and len(bbox) >= 4:
                                    # bbox is [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                                    min_x = min(point[0] for point in bbox)
                                    for num in numbers_in_text:
                                        numbers_with_positions.append((int(num), min_x, bbox))
                
                # --- FORMAT 3: Flat List (Rare edge case) ---
                elif isinstance(results, list) and len(results) > 0:
                    # Sometimes results is just the inner list directly
                    for line in results:
                        if isinstance(line, list) and len(line) >= 2 and isinstance(line[1], tuple):
                            bbox = line[0]
                            text = line[1][0]
                            score = line[1][1]
                            if score * 100 >= Config.OCR_CONFIDENCE_THRESHOLD:
                                numbers_in_text = re.findall(r'\d+', text)
                                if bbox and len(bbox) >= 4:
                                    min_x = min(point[0] for point in bbox)
                                    for num in numbers_in_text:
                                        numbers_with_positions.append((int(num), min_x, bbox))

            except Exception as ocr_e:
                print(f"  Internal OCR parsing error: {ocr_e}")
                return []
        
        # Sort by x-coordinate (position) and return the leftmost number with bbox
        if numbers_with_positions:
            try:
                # Sort by x-coordinate (second element), get the leftmost
                leftmost_entry = sorted(numbers_with_positions, key=lambda x: x[1])[0]
                leftmost_number = leftmost_entry[0]
                leftmost_bbox = leftmost_entry[2] if len(leftmost_entry) > 2 else None
                
                print(f"  OCR Found Leftmost Number: {leftmost_number} at x={leftmost_entry[1]} in {os.path.basename(image_path)}")
                print(f"  Bbox info available: {leftmost_bbox is not None}")
                if leftmost_bbox:
                    print(f"  Bbox coordinates: {leftmost_bbox}")
                    
                return [leftmost_number], leftmost_bbox
            except (ValueError, IndexError):
                pass
        
        print(f"  No OCR numbers found in {os.path.basename(image_path)}")
        return [], None
        
    except Exception as e:
        print(f"OCR error for {image_path}: {e}")
        return []
    
def compare_ocr_results(numbers1, numbers2, tolerance=0.1):
    """
    Compare two lists of OCR-extracted leftmost numbers.
    
    Args:
        numbers1: List containing leftmost number from first image (0-1 items)
        numbers2: List containing leftmost number from second image (0-1 items)
        tolerance: Tolerance for floating point comparison
    
    Returns:
        bool: True if numbers are significantly different (NOT duplicates)
    """
    # If both lists are empty, could be duplicates (no numbers found in either)
    if not numbers1 and not numbers2:
        return False  # Both empty, could be duplicates
    
    # If one has a number and the other doesn't, they're different
    if not numbers1 or not numbers2:
        return True   # One empty, one not - different
    
    # Both have exactly one number each (leftmost), compare them
    if len(numbers1) == 1 and len(numbers2) == 1:
        return abs(numbers1[0] - numbers2[0]) > tolerance
    
    # Shouldn't happen with the new implementation, but handle edge case
    return True  # Different if unexpected structure

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

def detect_frame_change(frame1, frame2, top_ratio=None, threshold=None):
    """
    Calculate the percentage change between two frames in the top portion.
    
    Args:
        frame1: Previous frame (numpy array)
        frame2: Current frame (numpy array)
        top_ratio: Ratio of top portion to analyze (default: Config.TOP_ANALYSIS_RATIO = 20%)
        threshold: Not used - kept for compatibility
    
    Returns:
        float: Percentage of changed pixels (0-100)
    """
    if top_ratio is None:
        top_ratio = Config.TOP_ANALYSIS_RATIO
    if threshold is None:
        threshold = Config.CHANGE_DETECTION_THRESHOLD
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
    
    # Threshold the difference (pixels with change > intensity threshold)
    _, thresh = cv2.threshold(diff, Config.PIXEL_INTENSITY_THRESHOLD, 255, cv2.THRESH_BINARY)
    
    # Calculate percentage of changed pixels
    total_pixels = top_height * width
    changed_pixels = cv2.countNonZero(thresh)
    change_percentage = (changed_pixels / total_pixels) * 100  # Return as percentage (0-100)
    
    return change_percentage

def extract_screenshots(video_path, start_time=2, interval=12, detection_method='time', change_threshold=None, force_recapture=False, custom_folder_name=None, disable_ocr=False, disable_duplicate_detection=False):
    """
    Extract screenshots from video with A/B capture system and organized folder structure.
    
    Args:
        video_path (str): Path to the video file
        start_time (int): Starting time in seconds (default: 2)
        interval (int): Interval between screenshots in seconds for time-based method (default: 12)
        detection_method (str): 'time' for fixed intervals, 'change' for content change detection
        change_threshold (float): Threshold for change detection (default: Config.CHANGE_DETECTION_THRESHOLD = 4%)
        force_recapture (bool): Force recapture even if raw folder exists
    """
    if change_threshold is None:
        change_threshold = Config.CHANGE_DETECTION_THRESHOLD
        
    # Check if video file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found.")
        return False
    
    # Get video filename without extension and sanitize it
    video_name = Path(video_path).stem
    sanitized_name = sanitize_filename(video_name)
    
    # Use custom folder name if provided, otherwise use sanitized video name
    if custom_folder_name:
        display_folder_name = sanitize_filename(custom_folder_name)
        # Create ASCII-safe folder name for file operations (OpenCV cannot write to Unicode paths on Windows)
        ascii_folder_name = sanitized_name  # Use original video name which is ASCII-safe
    else:
        display_folder_name = sanitized_name
        ascii_folder_name = sanitized_name
    
    print(f"Original filename: {video_name}")
    print(f"Display folder name: {display_folder_name}")
    print(f"ASCII folder name (for file ops): {ascii_folder_name}")
    
    # Create organized folder structure using ASCII-safe name in scores directory:
    # scores/[ascii_folder_name]/
    #   ├── screenshots/
    #   │   ├── raw/           # All A and B screenshots
    #   │   ├── result/        # Merged unique A+B screenshots
    #   │   └── duplicate/     # Duplicate pairs
    #   ├── [sanitized_name]_log.txt
    #   ├── [sanitized_name]_similarity_heatmap.html
    #   └── source_video.mp4   # Copy of original video
    scores_dir = os.path.join(os.path.dirname(os.getcwd()), "scores")
    os.makedirs(scores_dir, exist_ok=True)
    main_folder = os.path.join(scores_dir, ascii_folder_name)
    screenshots_dir = os.path.join(main_folder, "screenshots")
    raw_dir = os.path.join(screenshots_dir, "raw")
    result_dir = os.path.join(screenshots_dir, "result")
    duplicate_dir = os.path.join(screenshots_dir, "duplicate")
    
    # Check if we should skip recapture
    should_capture = True
    if not force_recapture and os.path.exists(raw_dir):
        # Check if raw folder has content
        existing_files = [f for f in os.listdir(raw_dir) if f.lower().endswith('.jpg')]
        if len(existing_files) > 0:
            should_capture = False
            print(f"Raw folder exists with {len(existing_files)} screenshots. Skipping capture.")
            print("Use --recapture flag to force recapture.")
    
    if should_capture:
        # Delete existing folder if force recapture
        if os.path.exists(main_folder) and force_recapture:
            import shutil
            print(f"Force recapture enabled. Deleting existing folder: {main_folder}")
            shutil.rmtree(main_folder)
            print("✓ Existing folder deleted")
        
        # Create new folder structure
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(result_dir, exist_ok=True)
        os.makedirs(duplicate_dir, exist_ok=True)
        print(f"Created main folder: {main_folder}")
        print(f"Raw screenshots will be saved to: {raw_dir}")
        
        # Copy source video to the score folder
        video_extension = os.path.splitext(video_path)[1]
        video_copy_path = os.path.join(main_folder, f"source_video{video_extension}")
        try:
            shutil.copy2(video_path, video_copy_path)
            print(f"✓ Source video copied to: {video_copy_path}")
        except Exception as e:
            print(f"⚠ Warning: Could not copy source video: {e}")
    else:
        # Ensure result and duplicate folders exist for processing
        os.makedirs(result_dir, exist_ok=True)
        os.makedirs(duplicate_dir, exist_ok=True)
        
        # Copy source video if it doesn't exist
        video_extension = os.path.splitext(video_path)[1]
        video_copy_path = os.path.join(main_folder, f"source_video{video_extension}")
        if not os.path.exists(video_copy_path):
            try:
                import shutil
                shutil.copy2(video_path, video_copy_path)
                print(f"✓ Source video copied to: {video_copy_path}")
            except Exception as e:
                print(f"⚠ Warning: Could not copy source video: {e}")
        else:
            print(f"Source video already exists: {video_copy_path}")
    
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
    
    if should_capture:
        print(f"Using {detection_method}-based method for A/B screenshot capture")
        if detection_method == 'time':
            print(f"Screenshots every {interval} seconds starting at {start_time}s")
            success = _extract_ab_time_based(cap, fps, duration, start_time, interval, raw_dir, log_path, sanitized_name)
        else:
            print(f"Detecting {change_threshold*100}% change in top 20% of frame")
            success = _extract_ab_change_based(cap, fps, duration, start_time, change_threshold, raw_dir, log_path, sanitized_name)
        
        cap.release()
        
        if not success:
            return False
    
    # Always run duplicate detection and processing steps
    print("\n=== PROCESSING EXISTING SCREENSHOTS ===")
    result = _process_screenshots(main_folder, raw_dir, result_dir, duplicate_dir, sanitized_name, disable_ocr=disable_ocr, disable_duplicate_detection=disable_duplicate_detection)
    
    # Return both success status and folder names for PDF creation
    return result, ascii_folder_name, display_folder_name

def _extract_ab_time_based(cap, fps, duration, start_time, interval, raw_dir, log_path, sanitized_name):
    """Extract A/B screenshots at fixed time intervals with 2-second B delay"""
    import time
    
    current_time = start_time
    process_start_time = time.time()
    screenshot_pairs = []  # Store valid A times for B capture
    
    # Initialize log file
    with open(log_path, 'w', encoding='utf-8') as log_file:
        log_file.write(f"Video Analysis Log - {sanitized_name}\n")
        log_file.write(f"Video Duration: {duration:.2f} seconds\n")
        log_file.write(f"Method: Time-based A/B capture (every {interval}s, B at +2s)\n")
        log_file.write(f"Start Time: {start_time}s\n")
        log_file.write("=" * 50 + "\n\n")
    
    screenshot_count = 0
    
    # Phase 1: Capture A screenshots and validate B timing
    print("Phase 1: Capturing A screenshots...")
    while current_time < duration:
        b_time = current_time + 2.0  # B screenshot 2 seconds later
        
        # Check if A screenshot is too close to video end (apply VIDEO_END_BUFFER rule)
        if current_time > (duration - Config.VIDEO_END_BUFFER):
            with open(log_path, 'a', encoding='utf-8') as log_file:
                log_file.write(f"SKIPPED A at {current_time:.1f}s (within {Config.VIDEO_END_BUFFER}s buffer of video end at {duration:.1f}s)\n")
            current_time += interval
            continue
        
        if b_time < duration:  # Only capture A if B is possible
            # Capture A screenshot
            frame_number = int(current_time * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            
            if ret:
                a_filename = f"{screenshot_count:02d}_{format_time(current_time)}_A.jpg"
                a_path = os.path.join(raw_dir, a_filename)
                
                if cv2.imwrite(a_path, frame):
                    screenshot_pairs.append((screenshot_count, current_time, b_time))
                    screenshot_count += 1
                    
                    with open(log_path, 'a', encoding='utf-8') as log_file:
                        log_file.write(f"A{screenshot_count:02d}: {current_time:.1f}s -> {a_filename}\n")
                        
                    print(f"  A{screenshot_count:02d} captured at {current_time:.1f}s")
        else:
            with open(log_path, 'a', encoding='utf-8') as log_file:
                log_file.write(f"SKIPPED A at {current_time:.1f}s (B would be at {b_time:.1f}s > {duration:.1f}s)\n")
        
        current_time += interval
    
    # Phase 2: Capture B screenshots
    print(f"Phase 2: Capturing {len(screenshot_pairs)} B screenshots...")
    b_captured = 0
    
    for count, a_time, b_time in screenshot_pairs:
        frame_number = int(b_time * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        
        if ret:
            b_filename = f"{count:02d}_{format_time(b_time)}_B.jpg"
            b_path = os.path.join(raw_dir, b_filename)
            
            if cv2.imwrite(b_path, frame):
                b_captured += 1
                with open(log_path, 'a', encoding='utf-8') as log_file:
                    log_file.write(f"B{count:02d}: {b_time:.1f}s -> {b_filename}\n")
                print(f"  B{count:02d} captured at {b_time:.1f}s")
    
    with open(log_path, 'a', encoding='utf-8') as log_file:
        log_file.write(f"\n=== CAPTURE COMPLETE ===\n")
        log_file.write(f"A screenshots: {len(screenshot_pairs)}\n")
        log_file.write(f"B screenshots: {b_captured}\n")
    
    print(f"✓ Captured {len(screenshot_pairs)} A screenshots and {b_captured} B screenshots")
    return True

def _extract_ab_change_based(cap, fps, duration, start_time, change_threshold, raw_dir, log_path, sanitized_name):
    """Extract A/B screenshots based on content changes with 2-second B delay"""
    import time
    
    # Start from the beginning or specified start time
    cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
    
    previous_frame = None
    frame_count = 0
    min_interval_frames = int(fps * Config.MIN_SCREENSHOT_INTERVAL)  # Minimum interval between A screenshots
    frames_since_last_screenshot = 0
    check_interval_frames = max(1, int(fps * Config.FRAME_CHECK_INTERVAL))  # Check frames interval
    screenshot_pairs = []  # Store valid A times for B capture
    
    print(f"Analyzing video for content changes (threshold: {change_threshold*100}%)...")
    print(f"Checking frames every {Config.FRAME_CHECK_INTERVAL}s ({check_interval_frames} frames)")
    
    # Initialize log file
    with open(log_path, 'w', encoding='utf-8') as log_file:
        log_file.write(f"Video Analysis Log - {sanitized_name}\n")
        log_file.write(f"Video Duration: {duration:.2f} seconds\n")
        log_file.write(f"Method: Change-based A/B capture ({change_threshold*100}% threshold, A at +{Config.A_CAPTURE_DELAY}s, B at +{Config.A_CAPTURE_DELAY + Config.B_CAPTURE_DELAY}s)\n")
        log_file.write(f"Start Time: {start_time}s\n")
        log_file.write("=" * 50 + "\n\n")
    
    # Progress tracking
    process_start_time = time.time()
    last_progress_update = 0
    screenshot_count = 0
    
    # Frame change tracking for visualization
    frame_change_data = []  # List of (time, change_percentage) tuples
    
    # Phase 1: Analyze and capture A screenshots
    print("Phase 1: Analyzing content changes and capturing A screenshots...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        frames_since_last_screenshot += 1
        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        
        # Update progress indicator
        current_real_time = time.time()
        if current_real_time - last_progress_update >= Config.PROGRESS_UPDATE_INTERVAL:
            progress_percent = (current_time / duration) * 100
            print(f"\rProgress: [{'█' * int(progress_percent/2.5)}{'.' * (40-int(progress_percent/2.5))}] {progress_percent:.1f}% ({current_time:.1f}s/{duration:.1f}s)", end='', flush=True)
            last_progress_update = current_real_time
        
        # Check for content changes
        if frame_count % check_interval_frames == 0:
            if previous_frame is not None:
                change_percentage = detect_frame_change(previous_frame, frame, top_ratio=Config.TOP_ANALYSIS_RATIO)
                
                # Store frame change data for visualization
                frame_change_data.append((current_time, change_percentage))
                
                # Check for major scene change
                if change_percentage >= Config.MAJOR_SCENE_CHANGE_THRESHOLD:
                    with open(log_path, 'a', encoding='utf-8') as log_file:
                        log_file.write(f"\n>>> MAJOR SCENE CHANGE DETECTED at {current_time:.1f}s ({change_percentage:.1f}% >= {Config.MAJOR_SCENE_CHANGE_THRESHOLD}%)\n")
                        log_file.write(f">>> STOPPING CAPTURE PROCESS\n")
                    print(f"\n\n>>> MAJOR SCENE CHANGE DETECTED at {current_time:.1f}s ({change_percentage:.1f}% >= {Config.MAJOR_SCENE_CHANGE_THRESHOLD}%)")
                    print(">>> STOPPING CAPTURE PROCESS")
                    break
                
                # Check if change exceeds threshold and minimum interval has passed
                if change_percentage >= (change_threshold * 100) and frames_since_last_screenshot >= min_interval_frames:
                    # Calculate delayed capture times
                    a_capture_time = current_time + Config.A_CAPTURE_DELAY  # Delay A capture by 0.5s
                    b_capture_time = a_capture_time + Config.B_CAPTURE_DELAY  # B capture after A
                    
                    # Check if delayed A screenshot is too close to video end
                    if a_capture_time > (duration - Config.VIDEO_END_BUFFER):
                        with open(log_path, 'a', encoding='utf-8') as log_file:
                            log_file.write(f"\nSKIPPED A at {current_time:.1f}s (delayed A would be at {a_capture_time:.1f}s within {Config.VIDEO_END_BUFFER}s buffer of video end at {duration:.1f}s)\n")
                    else:
                        # Check if B capture time is within video bounds
                        if b_capture_time < duration:
                            # Seek to delayed A capture time and capture
                            cap.set(cv2.CAP_PROP_POS_MSEC, a_capture_time * 1000)
                            ret_delayed, delayed_frame = cap.read()
                            
                            if ret_delayed:
                                a_filename = f"{screenshot_count:02d}_{format_time(a_capture_time)}_A.jpg"
                                a_path = os.path.join(raw_dir, a_filename)
                                
                                if cv2.imwrite(a_path, delayed_frame):
                                    screenshot_pairs.append((screenshot_count, a_capture_time, b_capture_time))
                                    screenshot_count += 1
                                    frames_since_last_screenshot = 0
                                    
                                    with open(log_path, 'a', encoding='utf-8') as log_file:
                                        log_file.write(f"\nCHANGE DETECTED at {current_time:.1f}s: {change_percentage:.1f}% change\n")
                                        log_file.write(f"  → A{screenshot_count:02d} captured at {a_capture_time:.1f}s (+{Config.A_CAPTURE_DELAY}s delay) -> {a_filename}\n")
                                        log_file.write(f"  → B scheduled for {b_capture_time:.1f}s (+{Config.A_CAPTURE_DELAY + Config.B_CAPTURE_DELAY}s total delay)\n")
                                    
                                    # Return to original position for continued analysis
                                    cap.set(cv2.CAP_PROP_POS_MSEC, current_time * 1000)
                            else:
                                with open(log_path, 'a', encoding='utf-8') as log_file:
                                    log_file.write(f"\nFAILED to read delayed A frame at {a_capture_time:.1f}s\n")
                                # Return to original position
                                cap.set(cv2.CAP_PROP_POS_MSEC, current_time * 1000)
                        else:
                            with open(log_path, 'a', encoding='utf-8') as log_file:
                                log_file.write(f"\nSKIPPED A at {current_time:.1f}s (delayed B would be at {b_capture_time:.1f}s > {duration:.1f}s)\n")
            else:
                # Save first frame with delayed timing if B is possible and not within video end buffer
                a_capture_time = current_time + Config.A_CAPTURE_DELAY
                b_capture_time = a_capture_time + Config.B_CAPTURE_DELAY
                
                if a_capture_time <= (duration - Config.VIDEO_END_BUFFER):
                    if b_capture_time < duration:
                        # Seek to delayed A capture time
                        cap.set(cv2.CAP_PROP_POS_MSEC, a_capture_time * 1000)
                        ret_delayed, delayed_frame = cap.read()
                        
                        if ret_delayed:
                            a_filename = f"{screenshot_count:02d}_{format_time(a_capture_time)}_A.jpg"
                            a_path = os.path.join(raw_dir, a_filename)
                            
                            if cv2.imwrite(a_path, delayed_frame):
                                screenshot_pairs.append((screenshot_count, a_capture_time, b_capture_time))
                                screenshot_count += 1
                                frames_since_last_screenshot = 0
                                
                                with open(log_path, 'a', encoding='utf-8') as log_file:
                                    log_file.write(f"\nINITIAL A{screenshot_count:02d}: {a_capture_time:.1f}s (+{Config.A_CAPTURE_DELAY}s delay) -> {a_filename}\n")
                                
                                # Return to original position for continued analysis
                                cap.set(cv2.CAP_PROP_POS_MSEC, current_time * 1000)
                else:
                    with open(log_path, 'a', encoding='utf-8') as log_file:
                        log_file.write(f"\nSKIPPED INITIAL A at {current_time:.1f}s (delayed A would be at {a_capture_time:.1f}s within {Config.VIDEO_END_BUFFER}s buffer of video end at {duration:.1f}s)\n")
        
        # Update previous frame every check interval
        if frame_count % check_interval_frames == 0:
            previous_frame = frame.copy()
    
    print(f"\n\nPhase 2: Capturing {len(screenshot_pairs)} B screenshots...")
    
    # Phase 2: Capture B screenshots
    b_captured = 0
    for count, a_time, b_time in screenshot_pairs:
        cap.set(cv2.CAP_PROP_POS_MSEC, b_time * 1000)
        ret, frame = cap.read()
        
        if ret:
            b_filename = f"{count:02d}_{format_time(b_time)}_B.jpg"
            b_path = os.path.join(raw_dir, b_filename)
            
            if cv2.imwrite(b_path, frame):
                b_captured += 1
                with open(log_path, 'a', encoding='utf-8') as log_file:
                    log_file.write(f"B{count:02d}: {b_time:.1f}s -> {b_filename}\n")
    
    with open(log_path, 'a', encoding='utf-8') as log_file:
        log_file.write(f"\n=== CAPTURE COMPLETE ===\n")
        log_file.write(f"A screenshots: {len(screenshot_pairs)}\n")
        log_file.write(f"B screenshots: {b_captured}\n")
    
    # Save frame change data for visualization
    frame_change_file = log_path.replace('_log.txt', '_frame_changes.txt')
    with open(frame_change_file, 'w', encoding='utf-8') as f:
        f.write("Time(s),ChangePercentage\n")
        for time_val, change_val in frame_change_data:
            f.write(f"{time_val:.2f},{change_val:.2f}\n")
    
    print(f"✓ Captured {len(screenshot_pairs)} A screenshots and {b_captured} B screenshots")
    print(f"✓ Frame change data saved: {frame_change_file}")
    return True

def format_time(seconds):
    """Format time as MM:SS for filenames"""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}m{secs:02d}s"

def _process_screenshots(main_folder, raw_dir, result_dir, duplicate_dir, sanitized_name, disable_ocr=False, disable_duplicate_detection=False):
    """Process screenshots: duplicate detection -> merging -> heatmap generation"""
    
    log_path = os.path.join(main_folder, f"{sanitized_name}_log.txt")
    
    if disable_duplicate_detection:
        print("=== DUPLICATE DETECTION SKIPPED ===")
        print("All duplicate detection disabled - processing all screenshots as unique")
        
        # Create empty duplicates list since no detection was performed
        duplicates = []
        
        # Log that duplicate detection was skipped
        with open(log_path, 'w', encoding='utf-8') as log_file:
            log_file.write(f"=== DUPLICATE DETECTION DISABLED ===\n")
            log_file.write(f"All duplicate detection tests skipped for faster processing\n")
            log_file.write(f"All screenshots will be processed as unique\n\n")
    else:
        print("=== DUPLICATE DETECTION ===")
        # Step 1: Duplicate detection on A screenshots only
        duplicates = detect_duplicates_advanced(raw_dir, duplicate_dir, log_path, disable_ocr=disable_ocr)
    
    print("=== MERGING UNIQUE SCREENSHOTS ===")
    # Step 2: Merge unique A screenshots with their B counterparts
    merge_results = merge_unique_ab_screenshots(raw_dir, result_dir, duplicates, log_path)
    
    print("=== GENERATING SIMILARITY HEATMAPS ===")
    # Step 3: Generate similarity heatmaps
    heatmap_path = os.path.join(main_folder, f"{sanitized_name}_similarity_heatmap.html")
    generate_triple_heatmap(raw_dir, duplicates, heatmap_path, sanitized_name)
    
    print(f"✓ Processing complete!")
    print(f"  - Duplicates detected: {len(duplicates)} pairs")
    print(f"  - Unique screenshots merged: {merge_results}")
    print(f"  - Heatmap saved: {heatmap_path}")
    
    return True, result_dir

def create_ocr_visual(image_path, ocr_numbers, bbox_info, output_path):
    """
    Create a visual representation of OCR results on the image.
    Shows the crop region and detected numbers with bounding boxes.
    """
    try:
        # Load original image
        img = cv2.imread(str(image_path))
        if img is None:
            return False
        
        height, width = img.shape[:2]
        
        # Draw OCR crop region (top portion and configurable horizontal ratio)
        crop_height = int(height * Config.OCR_CROP_RATIO)
        crop_width = int(width * Config.OCR_HORIZONTAL_RATIO)
        
        # Draw green rectangle for OCR region
        cv2.rectangle(img, (0, 0), (crop_width, crop_height), (0, 255, 0), 3)
        
        # Draw bounding box around detected number if available
        if bbox_info and ocr_numbers:
            print(f"  Drawing bbox for {ocr_numbers[0]} with bbox: {bbox_info}")
            try:
                # Convert bbox coordinates to integers and draw rectangle
                bbox_points = [[int(p[0]), int(p[1])] for p in bbox_info]
                bbox_array = np.array(bbox_points, np.int32)
                
                # Draw a thick yellow polygon around the detected number
                cv2.polylines(img, [bbox_array], True, (0, 255, 255), 3)  # Yellow box for detected number
                
                # Also draw a filled semi-transparent rectangle for better visibility
                overlay = img.copy()
                cv2.fillPoly(overlay, [bbox_array], (0, 255, 255))
                img = cv2.addWeighted(img, 0.8, overlay, 0.2, 0)  # 20% transparency
                
                # Add small text label at bbox with background
                min_x = min(p[0] for p in bbox_points)
                min_y = min(p[1] for p in bbox_points)
                max_x = max(p[0] for p in bbox_points)
                max_y = max(p[1] for p in bbox_points)
                
                # Draw text background rectangle
                text = f"DETECTED: {ocr_numbers[0]}"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.rectangle(img, (min_x-5, min_y-text_size[1]-10), 
                             (min_x+text_size[0]+5, min_y-5), (0, 0, 0), -1)  # Black background
                cv2.putText(img, text, (min_x, min_y-8), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)  # Yellow text
                
                # Draw crosshair at center of detected number
                center_x = (min_x + max_x) // 2
                center_y = (min_y + max_y) // 2
                cv2.drawMarker(img, (center_x, center_y), (255, 0, 0), cv2.MARKER_CROSS, 20, 3)  # Blue crosshair
                
            except Exception as e:
                print(f"Error drawing bbox: {e}")
        else:
            print(f"  No bbox to draw - bbox_info: {bbox_info is not None}, ocr_numbers: {ocr_numbers}")
        
        # Add text annotation for OCR region
        cv2.putText(img, f"OCR Region: {Config.OCR_CROP_RATIO*100:.0f}% top x {Config.OCR_HORIZONTAL_RATIO*100:.0f}% left", 
                   (10, height-60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add OCR result text
        ocr_text = f"OCR Result: {ocr_numbers if ocr_numbers else 'No numbers detected'}"
        cv2.putText(img, ocr_text, 
                   (10, height-30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Add filename at top with black text for better visibility
        filename = Path(image_path).name
        cv2.putText(img, filename, 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        # Save annotated image
        cv2.imwrite(str(output_path), img)
        return True
        
    except Exception as e:
        print(f"Error creating OCR visual for {image_path}: {e}")
        return False

def detect_duplicates_advanced(raw_dir, duplicate_dir, log_path, disable_ocr=False):
    """
    Advanced duplicate detection with two tests:
    1. Pixel-wise comparison of top portion
    2. Row-wise similarity analysis
    3. OCR number comparison (only if tests 1 or 2 pass, and if not disabled)
    
    Args:
        disable_ocr: If True, skip OCR testing for faster processing
    
    Returns list of duplicate pairs with their test results.
    """
    from pathlib import Path
    import numpy as np
    import shutil
    
    # Get all A screenshots
    a_files = sorted([f for f in Path(raw_dir).glob("*_A.jpg")])
    
    if len(a_files) <= 1:
        return []
    
    print(f"Analyzing {len(a_files)} A screenshots for duplicates...")
    
    # Create OCR result folder (clear if exists)
    main_folder = Path(raw_dir).parent.parent
    ocr_result_dir = main_folder / "ocr_result"
    if ocr_result_dir.exists():
        shutil.rmtree(ocr_result_dir)
    ocr_result_dir.mkdir(exist_ok=True)
    
    duplicates = []
    top_ratio = Config.DUPLICATE_TOP_RATIO  # Compare top portion only
    
    with open(log_path, 'a', encoding='utf-8') as log_file:
        log_file.write(f"\n=== ADVANCED DUPLICATE DETECTION ===\n")
        log_file.write(f"Screenshots to analyze: {len(a_files)}\n")
        log_file.write(f"Analysis region: Top {Config.DUPLICATE_TOP_RATIO*100}% of each image\n")
        log_file.write(f"Test 1: {Config.PIXEL_SIMILARITY_THRESHOLD*100}% pixel similarity threshold\n")
        log_file.write(f"Test 2: {Config.ROW_COVERAGE_THRESHOLD*100}% row similarity ({Config.ROW_SIMILARITY_THRESHOLD*100}% threshold per row)\n")
        log_file.write(f"Test 3: OCR number comparison (top {Config.OCR_CROP_RATIO*100}% of image)\n\n")
    
    # Compare each pair of A screenshots
    for i in range(len(a_files)):
        for j in range(i + 1, len(a_files)):
            file1, file2 = a_files[i], a_files[j]
            
            # Load images and extract top portions
            img1 = cv2.imread(str(file1))
            img2 = cv2.imread(str(file2))
            
            if img1 is None or img2 is None:
                continue
            
            # Extract top 27% portions
            h1, w1 = img1.shape[:2]
            h2, w2 = img2.shape[:2]
            
            top_h1 = int(h1 * top_ratio)
            top_h2 = int(h2 * top_ratio)
            
            top1 = img1[:top_h1, :]
            top2 = img2[:top_h2, :]
            
            # Resize to same dimensions if needed
            if top1.shape != top2.shape:
                min_h = min(top_h1, top_h2)
                min_w = min(w1, w2)
                top1 = cv2.resize(top1, (min_w, min_h))
                top2 = cv2.resize(top2, (min_w, min_h))
            
            # Test 1: Pixel-wise comparison
            test1_passed = test_pixel_similarity(top1, top2, threshold=Config.PIXEL_SIMILARITY_THRESHOLD)
            
            # Test 2: Row-wise comparison
            test2_passed = test_row_similarity(top1, top2, row_threshold=Config.ROW_SIMILARITY_THRESHOLD, coverage_threshold=Config.ROW_COVERAGE_THRESHOLD)
            
            # Only perform OCR if test 1 or test 2 passed (optimization) and OCR is not disabled
            if (test1_passed or test2_passed) and not disable_ocr:
                # Test 3: OCR comparison - if numbers are different, NOT a duplicate
                ocr_numbers1, bbox1 = extract_ocr_numbers(str(file1))
                ocr_numbers2, bbox2 = extract_ocr_numbers(str(file2))
                
                # Print OCR results when tests pass
                print(f"  OCR Results: {file1.name} = {ocr_numbers1}, {file2.name} = {ocr_numbers2}")
                
                # Create visual OCR results for this pair
                pair_id = f"pair_{i:02d}_{j:02d}"
                ocr_visual_1 = ocr_result_dir / f"{pair_id}_{file1.stem}_ocr.jpg"
                ocr_visual_2 = ocr_result_dir / f"{pair_id}_{file2.stem}_ocr.jpg"
                
                create_ocr_visual(str(file1), ocr_numbers1, bbox1, str(ocr_visual_1))
                create_ocr_visual(str(file2), ocr_numbers2, bbox2, str(ocr_visual_2))
                
                ocr_different = compare_ocr_results(ocr_numbers1, ocr_numbers2)
                
                # Only consider it a duplicate if OCR numbers are NOT different
                if not ocr_different:
                    duplicates.append((i, j, file1.name, file2.name, test1_passed, test2_passed, ocr_different, ocr_numbers1, ocr_numbers2))
                    
                    # Move duplicate pair to duplicate folder
                    pair_name = f"duplicate_pair_{len(duplicates):03d}"
                    
                    # Copy both files to duplicate folder
                    shutil.copy2(file1, os.path.join(duplicate_dir, f"{pair_name}_A_{file1.stem}.jpg"))
                    shutil.copy2(file2, os.path.join(duplicate_dir, f"{pair_name}_B_{file2.stem}.jpg"))
                    
                    with open(log_path, 'a', encoding='utf-8') as log_file:
                        log_file.write(f"DUPLICATE: {file1.name} vs {file2.name}\n")
                        log_file.write(f"  Test 1 (pixel): {'PASS' if test1_passed else 'FAIL'}\n")
                        log_file.write(f"  Test 2 (row): {'PASS' if test2_passed else 'FAIL'}\n")
                        log_file.write(f"  Test 3 (OCR): {'SAME' if not ocr_different else 'DIFFERENT'}\n")
                        log_file.write(f"  OCR Numbers 1: {ocr_numbers1}\n")
                        log_file.write(f"  OCR Numbers 2: {ocr_numbers2}\n")
                        log_file.write(f"  Saved as: {pair_name}\n\n")
                    
                    # Enhanced terminal output showing which test detected the duplicate
                    test_info = ""
                    if test1_passed and test2_passed:
                        test_info = " (Test 1 + Test 2, OCR Same)"
                    elif test1_passed:
                        test_info = " (Test 1: Pixel, OCR Same)"
                    elif test2_passed:
                        test_info = " (Test 2: Row, OCR Same)"
                    
                    print(f"  Duplicate found: {file1.name} ↔ {file2.name}{test_info}")
            elif (test1_passed or test2_passed) and disable_ocr:
                # OCR disabled but pixel/row tests passed - consider as potential duplicate without OCR verification
                print(f"  OCR disabled - considering as duplicate based on pixel/row tests: {file1.name} ↔ {file2.name}")
                duplicates.append((i, j, file1.name, file2.name, test1_passed, test2_passed, False, [], []))
                
                # Move duplicate pair to duplicate folder
                pair_name = f"duplicate_pair_{len(duplicates):03d}"
                
                # Copy both files to duplicate folder
                shutil.copy2(file1, os.path.join(duplicate_dir, f"{pair_name}_A_{file1.stem}.jpg"))
                shutil.copy2(file2, os.path.join(duplicate_dir, f"{pair_name}_B_{file2.stem}.jpg"))
                
                with open(log_path, 'a', encoding='utf-8') as log_file:
                    log_file.write(f"DUPLICATE (OCR DISABLED): {file1.name} vs {file2.name}\n")
                    log_file.write(f"  Test 1 (pixel): {'PASS' if test1_passed else 'FAIL'}\n")
                    log_file.write(f"  Test 2 (row): {'PASS' if test2_passed else 'FAIL'}\n")
                    log_file.write(f"  Test 3 (OCR): DISABLED\n")
                    log_file.write(f"  Saved as: {pair_name}\n\n")
                
                # Enhanced terminal output showing which test detected the duplicate
                test_info = ""
                if test1_passed and test2_passed:
                    test_info = " (Test 1 + Test 2, OCR Disabled)"
                elif test1_passed:
                    test_info = " (Test 1: Pixel, OCR Disabled)"
                elif test2_passed:
                    test_info = " (Test 2: Row, OCR Disabled)"
                
                print(f"  Duplicate found: {file1.name} ↔ {file2.name}{test_info}")
            else:
                # Either no tests passed or OCR test prevented duplicate
                if test1_passed or test2_passed:
                    # Log when similarity tests pass but OCR prevents duplicate classification
                    with open(log_path, 'a', encoding='utf-8') as log_file:
                        log_file.write(f"NOT DUPLICATE (OCR DIFFERENT): {file1.name} vs {file2.name}\n")
                        log_file.write(f"  Test 1 (pixel): {'PASS' if test1_passed else 'FAIL'}\n")
                        log_file.write(f"  Test 2 (row): {'PASS' if test2_passed else 'FAIL'}\n")
                        log_file.write(f"  Test 3 (OCR): DIFFERENT\n")
                        log_file.write(f"  OCR Numbers 1: {ocr_numbers1 if not disable_ocr else 'N/A'}\n")
                        log_file.write(f"  OCR Numbers 2: {ocr_numbers2 if not disable_ocr else 'N/A'}\n\n")
                    
                    print(f"  Not duplicate (OCR differs): {file1.name} ↔ {file2.name} (OCR: {ocr_numbers1 if not disable_ocr else 'N/A'} vs {ocr_numbers2 if not disable_ocr else 'N/A'})")
    
    with open(log_path, 'a', encoding='utf-8') as log_file:
        log_file.write(f"=== DUPLICATE DETECTION COMPLETE ===\n")
        log_file.write(f"Total duplicates found: {len(duplicates)} pairs\n\n")
    
    return duplicates

def test_pixel_similarity(img1, img2, threshold=0.95):
    """Test 1: Pixel-wise comparison with 95% similarity threshold"""
    total_pixels = img1.size
    identical_pixels = np.sum(img1 == img2)
    similarity = identical_pixels / total_pixels
    return similarity >= threshold

def test_row_similarity(img1, img2, row_threshold=0.98, coverage_threshold=0.94):
    """Test 2: Row-wise comparison with 94% of rows being 98% similar"""
    if img1.shape != img2.shape:
        return False
    
    h, w, c = img1.shape
    similar_rows = 0
    
    for row in range(h):
        row1 = img1[row, :, :].flatten()
        row2 = img2[row, :, :].flatten()
        
        # Calculate row similarity using safer method
        # Convert to float64 to prevent overflow
        row_sum1 = np.sum(row1, dtype=np.float64)
        row_sum2 = np.sum(row2, dtype=np.float64)
        
        if row_sum1 == 0 and row_sum2 == 0:
            similar_rows += 1
        elif row_sum1 > 0 and row_sum2 > 0:
            # Use safer similarity calculation to prevent overflow
            max_sum = max(row_sum1, row_sum2)
            min_sum = min(row_sum1, row_sum2)
            similarity = min_sum / max_sum  # This gives ratio 0-1, higher = more similar
            if similarity >= row_threshold:
                similar_rows += 1
    
    coverage = similar_rows / h
    return coverage >= coverage_threshold

def merge_unique_ab_screenshots(raw_dir, result_dir, duplicates, log_path):
    """Merge unique A screenshots with their B counterparts (left 30% overlay)"""
    from pathlib import Path
    import shutil
    
    # Clear result directory to ensure clean merge
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.makedirs(result_dir, exist_ok=True)
    
    # Get all A screenshots
    a_files = sorted([f for f in Path(raw_dir).glob("*_A.jpg")])
    
    # Get list of duplicate A screenshot indices (mark LATER ones as duplicates)
    duplicate_indices = set()
    for dup in duplicates:
        # Keep the earlier one (smaller index), mark later one as duplicate
        duplicate_indices.add(max(dup[0], dup[1]))  # Mark the LATER one as duplicate
    
    print(f"Debug: Total A files: {len(a_files)}")
    print(f"Debug: Duplicate pairs: {len(duplicates)}")
    print(f"Debug: Duplicate indices to remove: {sorted(duplicate_indices)}")
    
    # Get unique indices (all indices EXCEPT the later duplicates)
    unique_indices = []
    for i in range(len(a_files)):
        if i not in duplicate_indices:
            unique_indices.append(i)
    
    print(f"Debug: Unique indices remaining: {sorted(unique_indices)}")
    print(f"Merging {len(unique_indices)} unique A screenshots with their B counterparts...")
    
    merged_count = 0
    
    with open(log_path, 'a', encoding='utf-8') as log_file:
        log_file.write(f"=== MERGING UNIQUE A+B SCREENSHOTS ===\n")
        log_file.write(f"Total A screenshots: {len(a_files)}\n")
        log_file.write(f"Duplicate pairs found: {len(duplicates)}\n")
        log_file.write(f"Later duplicate indices: {sorted(duplicate_indices)}\n")
        log_file.write(f"Unique A screenshots to merge: {len(unique_indices)}\n\n")
    
    for sequential_idx, idx in enumerate(unique_indices):
        if idx >= len(a_files):
            continue
            
        a_file = a_files[idx]
        
        # Find corresponding B file by matching file index (not calculated timing)
        # A and B files from the same capture event have the same index
        import re
        time_match = re.search(r'(\d{2})_(\d{2})m(\d{2})s_A\.jpg', a_file.name)
        if time_match:
            file_idx = time_match.groups()[0]
            # Look for any B file with the same index
            b_pattern = f"{file_idx}_*_B.jpg"
            b_candidates = list(Path(raw_dir).glob(b_pattern))
            
            if b_candidates:
                b_file = b_candidates[0]  # Take the first (should be only) match
            else:
                with open(log_path, 'a', encoding='utf-8') as log_file:
                    log_file.write(f"WARNING: No B file found for {a_file.name} (searched pattern: {b_pattern})\n")
                print(f"DEBUG: Missing B file - A: {a_file.name}, Pattern: {b_pattern}")
                continue
        else:
            # Fallback to simple replacement if pattern doesn't match
            b_name = a_file.stem.replace('_A', '_B') + '.jpg'
            b_file = Path(raw_dir) / b_name
            
            if not b_file.exists():
                with open(log_path, 'a', encoding='utf-8') as log_file:
                    log_file.write(f"WARNING: No B file found for {a_file.name} (expected: {b_name})\n")
                continue
        
        # Load both images
        img_a = cv2.imread(str(a_file))
        img_b = cv2.imread(str(b_file))
        
        if img_a is None or img_b is None:
            continue
        
        # Ensure same dimensions
        h_a, w_a = img_a.shape[:2]
        h_b, w_b = img_b.shape[:2]
        
        if (h_a, w_a) != (h_b, w_b):
            img_b = cv2.resize(img_b, (w_a, h_a))
        
        # Take left portion of B and overlay on A
        left_width = int(w_a * Config.B_OVERLAY_WIDTH_RATIO)
        img_a[:, :left_width] = img_b[:, :left_width]
        
        # Save merged result
        result_filename = f"{sequential_idx:02d}_{a_file.stem.replace('_A', '')}_merged.jpg"
        result_path = os.path.join(result_dir, result_filename)
        
        if cv2.imwrite(result_path, img_a):
            merged_count += 1
            with open(log_path, 'a', encoding='utf-8') as log_file:
                log_file.write(f"MERGED: {a_file.name} + {b_file.name} -> {result_filename}\n")
    
    with open(log_path, 'a', encoding='utf-8') as log_file:
        log_file.write(f"\n=== MERGING COMPLETE ===\n")
        log_file.write(f"Successfully merged: {merged_count} screenshots\n\n")
    
    return merged_count

def generate_triple_heatmap(raw_dir, duplicates, heatmap_path, sanitized_name):
    """Generate comprehensive analysis dashboard: frame change plot, similarity heatmaps, and duplicate map"""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("Plotly not available. Skipping heatmap generation.")
        return
    
    from pathlib import Path
    
    # Get all A screenshots
    a_files = sorted([f for f in Path(raw_dir).glob("*_A.jpg")])
    n = len(a_files)
    
    if n <= 1:
        print("Not enough screenshots for heatmap generation.")
        return
    
    print(f"Generating comprehensive analysis dashboard for {n} screenshots...")
    
    # Load frame change data if available
    main_folder = Path(raw_dir).parent.parent
    frame_change_file = main_folder / f"{sanitized_name}_frame_changes.txt"
    frame_times = []
    frame_changes = []
    
    if frame_change_file.exists():
        try:
            with open(frame_change_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()[1:]  # Skip header
                for line in lines:
                    time_val, change_val = line.strip().split(',')
                    frame_times.append(float(time_val))
                    frame_changes.append(float(change_val))
            print(f"Loaded {len(frame_times)} frame change data points")
        except Exception as e:
            print(f"Could not load frame change data: {e}")
    
    # Initialize similarity matrices
    test1_matrix = np.zeros((n, n))
    test2_matrix = np.zeros((n, n))
    duplicate_matrix = np.zeros((n, n))
    
    # Fill diagonal with 1.0 (perfect similarity with self)
    np.fill_diagonal(test1_matrix, 1.0)
    np.fill_diagonal(test2_matrix, 1.0)
    np.fill_diagonal(duplicate_matrix, 1.0)
    
    top_ratio = Config.DUPLICATE_TOP_RATIO  # Top portion of images for analysis
    
    # Calculate all pairwise similarities
    for i in range(n):
        for j in range(i + 1, n):
            file1, file2 = a_files[i], a_files[j]
            
            # Load images and extract top portions
            img1 = cv2.imread(str(file1))
            img2 = cv2.imread(str(file2))
            
            if img1 is None or img2 is None:
                continue
            
            # Extract top 27% portions
            h1, w1 = img1.shape[:2]
            h2, w2 = img2.shape[:2]
            
            top_h1 = int(h1 * top_ratio)
            top_h2 = int(h2 * top_ratio)
            
            top1 = img1[:top_h1, :]
            top2 = img2[:top_h2, :]
            
            # Resize to same dimensions if needed
            if top1.shape != top2.shape:
                min_h = min(top_h1, top_h2)
                min_w = min(w1, w2)
                top1 = cv2.resize(top1, (min_w, min_h))
                top2 = cv2.resize(top2, (min_w, min_h))
            
            # Calculate Test 1 similarity (pixel-wise)
            total_pixels = top1.size
            identical_pixels = np.sum(top1 == top2)
            test1_similarity = identical_pixels / total_pixels
            
            # Calculate Test 2 similarity (row-wise)
            test2_similarity = calculate_row_similarity(top1, top2)
            
            # Check if it's a duplicate
            is_duplicate = (test1_similarity >= Config.PIXEL_SIMILARITY_THRESHOLD) or (test2_similarity >= Config.ROW_COVERAGE_THRESHOLD)
            
            # Fill matrices (symmetric)
            test1_matrix[i, j] = test1_matrix[j, i] = test1_similarity
            test2_matrix[i, j] = test2_matrix[j, i] = test2_similarity
            duplicate_matrix[i, j] = duplicate_matrix[j, i] = 1 if is_duplicate else 0
    
    # Create screenshot labels with better spacing
    labels = [f"{i:02d}_{f.stem.replace('_A', '')}" for i, f in enumerate(a_files)]
    
    # Create a 2x2 subplot layout for comprehensive dashboard
    if frame_times and frame_changes:
        # Create 2x2 subplot with frame change plot
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                f"Frame Change Analysis (every {Config.FRAME_CHECK_INTERVAL}s)",
                f"Test 1: Pixel Similarity (≥{Config.PIXEL_SIMILARITY_THRESHOLD*100}% = duplicate)",
                f"Test 2: Row Similarity (≥{Config.ROW_COVERAGE_THRESHOLD*100}% = duplicate)",
                f"Boolean Duplicate Map ({len(duplicates)} pairs found)"
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.08,
            specs=[[{"type": "scatter"}, {"type": "heatmap"}],
                   [{"type": "heatmap"}, {"type": "heatmap"}]]
        )
        
        # Frame change plot (top left)
        fig.add_trace(
            go.Scatter(
                x=frame_times,
                y=frame_changes,
                mode='lines+markers',
                name='Frame Change %',
                line=dict(width=2, color='blue'),
                marker=dict(size=4),
                hovertemplate='Time: %{x:.1f}s<br>Change: %{y:.2f}%<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add threshold lines to frame change plot
        fig.add_hline(y=Config.CHANGE_DETECTION_THRESHOLD*100, line_dash="dash", 
                     line_color="orange", annotation_text=f"Capture Threshold ({Config.CHANGE_DETECTION_THRESHOLD*100}%)")
        fig.add_hline(y=Config.MAJOR_SCENE_CHANGE_THRESHOLD, line_dash="dash", 
                     line_color="red", annotation_text=f"Major Change Threshold ({Config.MAJOR_SCENE_CHANGE_THRESHOLD}%)")
        
        # Add screenshot capture markers
        screenshot_times = []
        for a_file in a_files:
            time_match = re.search(r'(\d{2})m(\d{2})s', a_file.name)
            if time_match:
                minutes, seconds = time_match.groups()
                time_val = int(minutes) * 60 + int(seconds)
                screenshot_times.append(time_val)
        
        if screenshot_times:
            # Find corresponding change values for screenshot times
            screenshot_changes = []
            for st in screenshot_times:
                # Find closest frame time
                closest_idx = min(range(len(frame_times)), key=lambda i: abs(frame_times[i] - st))
                screenshot_changes.append(frame_changes[closest_idx])
            
            fig.add_trace(
                go.Scatter(
                    x=screenshot_times,
                    y=screenshot_changes,
                    mode='markers',
                    name='Screenshots Taken',
                    marker=dict(size=8, color='red', symbol='diamond'),
                    hovertemplate='Screenshot at %{x:.1f}s<br>Change: %{y:.2f}%<extra></extra>'
                ),
                row=1, col=1
            )
        
        # Test 1 heatmap (top right)
        fig.add_trace(
            go.Heatmap(
                z=test1_matrix,
                x=labels,
                y=labels,
                colorscale='RdYlBu_r',
                zmin=0,
                zmax=1,
                showscale=True,
                colorbar=dict(title="Pixel Similarity", x=0.95, len=0.4, y=0.8)
            ),
            row=1, col=2
        )
        
        # Test 2 heatmap (bottom left)
        fig.add_trace(
            go.Heatmap(
                z=test2_matrix,
                x=labels,
                y=labels,
                colorscale='RdYlBu_r',
                zmin=0,
                zmax=1,
                showscale=True,
                colorbar=dict(title="Row Similarity", x=0.48, len=0.4, y=0.2)
            ),
            row=2, col=1
        )
        
        # Boolean duplicate map (bottom right)
        fig.add_trace(
            go.Heatmap(
                z=duplicate_matrix,
                x=labels,
                y=labels,
                colorscale=[[0, 'white'], [1, 'red']],
                showscale=True,
                colorbar=dict(title="Duplicate", x=0.95, len=0.4, y=0.2)
            ),
            row=2, col=2
        )
        
        # Update layout for 2x2
        fig.update_layout(
            title=f"Comprehensive Video Analysis Dashboard - {sanitized_name}",
            height=1000,
            width=1400,
            font=dict(size=10),
            showlegend=True
        )
    else:
        # Fallback to 1x3 layout if no frame change data
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=[
                f"Test 1: Pixel Similarity (≥{Config.PIXEL_SIMILARITY_THRESHOLD*100}% = duplicate)",
                f"Test 2: Row Similarity (≥{Config.ROW_COVERAGE_THRESHOLD*100}% = duplicate)",
                f"Boolean Duplicate Map ({len(duplicates)} pairs found)"
            ],
            horizontal_spacing=0.08
        )
        
        # Test 1 heatmap
        fig.add_trace(
            go.Heatmap(
                z=test1_matrix,
                x=labels,
                y=labels,
                colorscale='RdYlBu_r',
                zmin=0,
                zmax=1,
                showscale=True,
                colorbar=dict(title="Pixel Similarity", x=0.35)
            ),
            row=1, col=1
        )
        
        # Test 2 heatmap
        fig.add_trace(
            go.Heatmap(
                z=test2_matrix,
                x=labels,
                y=labels,
                colorscale='RdYlBu_r',
                zmin=0,
                zmax=1,
                showscale=True,
                colorbar=dict(title="Row Similarity", x=0.68)
            ),
            row=1, col=2
        )
        
        # Boolean duplicate map
        fig.add_trace(
            go.Heatmap(
                z=duplicate_matrix,
                x=labels,
                y=labels,
                colorscale=[[0, 'white'], [1, 'red']],
                showscale=True,
                colorbar=dict(title="Duplicate", x=1.0)
            ),
            row=1, col=3
        )
        
        # Update layout for 1x3
        fig.update_layout(
            title=f"Screenshot Similarity Analysis - {sanitized_name}",
            height=600,
            width=1800,
            font=dict(size=10)
        )
    
    # Improve label spacing and readability for all heatmaps
    fig.update_xaxes(tickangle=45, tickfont=dict(size=8))
    fig.update_yaxes(tickfont=dict(size=8))
    
    # Add margin for better label visibility
    fig.update_layout(margin=dict(l=80, r=80, t=120, b=100))
    
    # Add comprehensive statistics annotation
    test1_count = np.sum(test1_matrix >= Config.PIXEL_SIMILARITY_THRESHOLD) // 2
    test2_count = np.sum(test2_matrix >= Config.ROW_COVERAGE_THRESHOLD) // 2
    total_pairs = n * (n - 1) // 2
    
    stats_text = (
        f"<b>Analysis Statistics:</b><br>"
        f"• Total screenshot pairs: {total_pairs}<br>"
        f"• Test 1 threshold hits: {test1_count}<br>"
        f"• Test 2 threshold hits: {test2_count}<br>"
        f"• Duplicates found: {len(duplicates)}<br>"
        f"• Unique screenshots: {n - len(set(dup[1] for dup in duplicates))}<br>"
    )
    
    if frame_times and frame_changes:
        avg_change = sum(frame_changes) / len(frame_changes)
        max_change = max(frame_changes)
        stats_text += (
            f"• Frame change points: {len(frame_times)}<br>"
            f"• Average change: {avg_change:.2f}%<br>"
            f"• Maximum change: {max_change:.2f}%"
        )
    
    fig.add_annotation(
        text=stats_text,
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        xanchor="left", yanchor="top",
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="black",
        borderwidth=1,
        font=dict(size=10)
    )
    
    # Save the comprehensive dashboard
    fig.write_html(heatmap_path)
    dashboard_type = "comprehensive dashboard" if frame_times and frame_changes else "similarity analysis"
    print(f"✓ {dashboard_type.title()} saved: {heatmap_path}")

def calculate_row_similarity(img1, img2):
    """Calculate overall row similarity percentage for Test 2"""
    if img1.shape != img2.shape:
        return 0.0
    
    h, w, c = img1.shape
    similar_rows = 0
    
    for row in range(h):
        row1 = img1[row, :, :].flatten()
        row2 = img2[row, :, :].flatten()
        
        # Calculate row similarity using safer method
        # Convert to float64 to prevent overflow
        row_sum1 = np.sum(row1, dtype=np.float64)
        row_sum2 = np.sum(row2, dtype=np.float64)
        
        if row_sum1 == 0 and row_sum2 == 0:
            similar_rows += 1
        elif row_sum1 > 0 and row_sum2 > 0:
            # Use safer similarity calculation to prevent overflow
            max_sum = max(row_sum1, row_sum2)
            min_sum = min(row_sum1, row_sum2)
            similarity = min_sum / max_sum  # This gives ratio 0-1, higher = more similar
            if similarity >= Config.ROW_SIMILARITY_THRESHOLD:  # Row similarity threshold
                similar_rows += 1
    
    return similar_rows / h

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
    removed_count = remove_duplicate_screenshots(screenshots_dir, log_path, top_ratio=Config.TOP_ANALYSIS_RATIO)
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
    check_interval_frames = max(1, int(fps * 1.0))  # Check every 1.0 second
    screenshot_times = []  # Track times when screenshots A were taken
    
    print(f"Analyzing video for content changes (threshold: {change_threshold*100}%)...")
    print(f"Checking frames every {1.0}s ({check_interval_frames} frames)")
    
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
                
                # Check if change is over 50% - abort if so
                if change_percentage >= 50.0:
                    with open(log_path, 'a', encoding='utf-8') as log_file:
                        log_file.write(f"\n>>> MAJOR SCENE CHANGE DETECTED at {current_time:.1f}s ({change_percentage:.1f}% >= 50%)\n")
                        log_file.write(f">>> ABORTING CAPTURE PROCESS\n")
                    print(f"\n\n>>> MAJOR SCENE CHANGE DETECTED at {current_time:.1f}s ({change_percentage:.1f}% >= 50%)")
                    print(">>> ABORTING CAPTURE PROCESS - generating PDF with current screenshots")
                    break
                
                # Check if change exceeds threshold and minimum interval has passed
                if change_percentage >= (change_threshold * 100) and frames_since_last_screenshot >= min_interval_frames:
                    with open(log_path, 'a', encoding='utf-8') as log_file:
                        log_file.write(f"\n>>> SCREENSHOT A CAPTURED at {current_time:.1f}s ({change_percentage:.1f}% >= {change_threshold*100}%)\n")
                    success = _save_screenshot(frame, current_time, screenshot_count, screenshots_dir)
                    if success:
                        screenshot_times.append(current_time)
                        screenshot_count += 1
                        frames_since_last_screenshot = 0
            else:
                # Save first frame after start time
                with open(log_path, 'a', encoding='utf-8') as log_file:
                    log_file.write(f"\n>>> INITIAL SCREENSHOT A CAPTURED at {current_time:.1f}s\n")
                success = _save_screenshot(frame, current_time, screenshot_count, screenshots_dir)
                if success:
                    screenshot_times.append(current_time)
                    screenshot_count += 1
                    frames_since_last_screenshot = 0
        
        # Update previous frame every 0.2 seconds for comparison
        if frame_count % check_interval_frames == 0:
            previous_frame = frame.copy()
    
    # Now capture the B screenshots (2.5 seconds after each A screenshot)
    print(f"\n\nCapturing B screenshots (2.5 seconds after each A screenshot)...")
    b_screenshot_count = 0
    
    for i, a_time in enumerate(screenshot_times):
        b_time = a_time + 2.5  # 2.5 seconds after A screenshot
        if b_time < duration:
            # Set video position to B time
            cap.set(cv2.CAP_PROP_POS_MSEC, b_time * 1000)
            ret, b_frame = cap.read()
            
            if ret:
                # Save B screenshot with special naming convention
                b_success = _save_screenshot_b(b_frame, b_time, i + 1, screenshots_dir)
                if b_success:
                    b_screenshot_count += 1
                    with open(log_path, 'a', encoding='utf-8') as log_file:
                        log_file.write(f">>> SCREENSHOT B CAPTURED at {b_time:.1f}s (for A screenshot {i+1})\n")
                else:
                    with open(log_path, 'a', encoding='utf-8') as log_file:
                        log_file.write(f">>> Failed to save B screenshot at {b_time:.1f}s\n")
            else:
                with open(log_path, 'a', encoding='utf-8') as log_file:
                    log_file.write(f">>> Could not read frame for B screenshot at {b_time:.1f}s\n")
    
    print(f"Captured {b_screenshot_count} B screenshots")
    
    # Release the video capture object now that we're done with B screenshots
    cap.release()
    
    # Final progress update
    print(f"\rProgress: [{'█' * 40}] 100.0% ({duration:.1f}s/{duration:.1f}s) - {screenshot_count} screenshots")
    
    # Final log entry
    with open(log_path, 'a', encoding='utf-8') as log_file:
        log_file.write(f"\n" + "=" * 50 + "\n")
        log_file.write(f"Analysis completed at {duration:.1f}s\n")
        log_file.write(f"Total A screenshots captured: {screenshot_count}\n")
        log_file.write(f"Total B screenshots captured: {b_screenshot_count}\n")
    
    # Remove duplicate screenshots BEFORE merging
    removed_count = remove_duplicate_screenshots(screenshots_dir, log_path, top_ratio=Config.TOP_ANALYSIS_RATIO)
    final_count = screenshot_count - removed_count
    
    print(f"\nDuplicate removal complete! Removed {removed_count} duplicates from {screenshot_count} screenshots.")
    print(f"Remaining screenshots for merging: {final_count}")
    
    # NOW merge B screenshots with remaining A screenshots and cleanup
    print("Merging B screenshots with remaining A screenshots...")
    _merge_b_to_a_screenshots(screenshots_dir, log_path)
    
    with open(log_path, 'a', encoding='utf-8') as log_file:
        log_file.write(f"B screenshots merged and deleted\n")
    
    print(f"\nCompleted! Extracted {screenshot_count} screenshots using change detection, removed {removed_count} duplicates.")
    print(f"Final count: {final_count} unique screenshots with merged progress bars.")
    print(f"Screenshots saved in: {screenshots_dir}")
    print(f"Log saved to: {log_path}")
    return True

def _calculate_frame_change_percentage(frame1, frame2, top_ratio=None):
    """
    Calculate the percentage of pixels that changed in the top portion of frames.
    
    Args:
        frame1: Previous frame (numpy array)
        frame2: Current frame (numpy array)
        top_ratio: Ratio of top portion to analyze (default: Config.TOP_ANALYSIS_RATIO = 20%)
    
    Returns:
        float: Percentage of changed pixels (0-100)
    """
    if top_ratio is None:
        top_ratio = Config.TOP_ANALYSIS_RATIO
        
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
    
    # Threshold the difference (pixels with change > intensity threshold)
    _, thresh = cv2.threshold(diff, Config.PIXEL_INTENSITY_THRESHOLD, 255, cv2.THRESH_BINARY)
    
    # Calculate percentage of changed pixels
    total_pixels = top_height * width
    changed_pixels = cv2.countNonZero(thresh)
    change_percentage = (changed_pixels / total_pixels) * 100
    
    return change_percentage

def remove_duplicate_screenshots(screenshots_dir, log_path, similarity_threshold=0.05, top_ratio=None):
    """
    Remove duplicate screenshots by comparing their similarity and save duplicate pairs.
    Only compares the top portion of the images as specified by top_ratio.
    Similar screenshots are saved as A/B pairs in a 'duplicates' folder.
    
    Args:
        screenshots_dir (str): Directory containing screenshots
        log_path (str): Path to log file
        similarity_threshold (float): Threshold for considering images as duplicates (0.0-1.0)
        top_ratio (float): Ratio of top portion to analyze (default: Config.TOP_ANALYSIS_RATIO = 20%)
        
    Returns:
        int: Number of duplicate screenshots removed from main folder
    """
    if top_ratio is None:
        top_ratio = Config.TOP_ANALYSIS_RATIO
        
    import hashlib
    from pathlib import Path
    import shutil
    
    # Create duplicates folder
    main_dir = Path(screenshots_dir).parent
    duplicates_dir = main_dir / "duplicates"
    duplicates_dir.mkdir(exist_ok=True)
    
    # Get all screenshot files - ONLY A screenshots, not B screenshots
    all_files = [f for f in Path(screenshots_dir).glob("*.jpg") if "screenshot_" in f.name]
    screenshot_files = [f for f in all_files if "_B_" not in f.name]  # Exclude B screenshots
    screenshot_files = sorted(screenshot_files, key=lambda x: int(x.name.split('_')[1]))  # Sort by sequence number
    
    # Log duplicate removal start
    with open(log_path, 'a', encoding='utf-8') as log_file:
        log_file.write(f"\n" + "=" * 50 + "\n")
        log_file.write(f"DUPLICATE REMOVAL ANALYSIS\n")
        log_file.write(f"=" * 50 + "\n")
        log_file.write(f"Total screenshots to analyze: {len(screenshot_files)}\n")
        log_file.write(f"Similarity threshold: {similarity_threshold} ({similarity_threshold*100}%)\n")
        log_file.write(f"Analysis region: Top {top_ratio*100}% of each image\n")
        log_file.write(f"Screenshots directory: {screenshots_dir}\n\n")
    
    if len(screenshot_files) <= 1:
        with open(log_path, 'a', encoding='utf-8') as log_file:
            log_file.write(f"Only {len(screenshot_files)} screenshot(s) found - no duplicates to remove\n")
        return 0
    
    removed_count = 0
    duplicate_pair_count = 0
    kept_files = [screenshot_files[0]]  # Always keep the first screenshot
    
    print(f"Checking {len(screenshot_files)} screenshots for duplicates...")
    
    # Log that we're keeping the first screenshot
    with open(log_path, 'a', encoding='utf-8') as log_file:
        log_file.write(f"KEPT: {screenshot_files[0].name} (first screenshot - always kept)\n")
        log_file.write(f"Duplicates will be saved to: {duplicates_dir}\n\n")
    
    for i in range(1, len(screenshot_files)):
        current_file = screenshot_files[i]
        is_duplicate = False
        
        # Compare with all previously kept files
        for kept_file in kept_files:
            similarity = are_images_similar(str(kept_file), str(current_file), similarity_threshold, top_ratio)
            if similarity is not None and similarity <= similarity_threshold:
                duplicate_pair_count += 1
                
                # Create A and B filenames for the duplicate pair
                base_name = f"duplicate_pair_{duplicate_pair_count:03d}"
                a_filename = f"{base_name}_A_{kept_file.stem}.jpg"
                b_filename = f"{base_name}_B_{current_file.stem}.jpg"
                
                # Copy both files to duplicates folder
                shutil.copy2(kept_file, duplicates_dir / a_filename)
                shutil.copy2(current_file, duplicates_dir / b_filename)
                
                print(f"  Found duplicate pair: {current_file.name} (similar to {kept_file.name}) - saved as {base_name}")
                with open(log_path, 'a', encoding='utf-8') as log_file:
                    log_file.write(f"DUPLICATE PAIR {duplicate_pair_count}: {kept_file.name} (A) vs {current_file.name} (B)\n")
                    log_file.write(f"  Similarity: {similarity:.4f} <= {similarity_threshold}\n")
                    log_file.write(f"  Saved as: {a_filename} and {b_filename}\n")
                
                # Remove the duplicate from main folder
                os.remove(current_file)
                removed_count += 1
                is_duplicate = True
                break
            elif similarity is not None:
                with open(log_path, 'a', encoding='utf-8') as log_file:
                    log_file.write(f"COMPARED: {current_file.name} vs {kept_file.name} (similarity: {similarity:.4f} > {similarity_threshold} - not duplicate)\n")
        
        if not is_duplicate:
            kept_files.append(current_file)
            with open(log_path, 'a', encoding='utf-8') as log_file:
                log_file.write(f"KEPT: {current_file.name} (unique screenshot)\n")
    
    print(f"Found {duplicate_pair_count} duplicate pairs, removed {removed_count} duplicates from main folder")
    
    # Log duplicate removal summary
    with open(log_path, 'a', encoding='utf-8') as log_file:
        log_file.write(f"\n" + "-" * 30 + "\n")
        log_file.write(f"DUPLICATE REMOVAL SUMMARY\n")
        log_file.write(f"-" * 30 + "\n")
        log_file.write(f"Total screenshots analyzed: {len(screenshot_files)}\n")
        log_file.write(f"Duplicate pairs found: {duplicate_pair_count}\n")
        log_file.write(f"Screenshots removed from main folder: {removed_count}\n")
        log_file.write(f"Screenshots kept in main folder: {len(kept_files)}\n")
        log_file.write(f"Duplicates saved to: {duplicates_dir}\n")
        log_file.write(f"Removal rate: {(removed_count/len(screenshot_files)*100):.1f}%\n")
        if kept_files:
            log_file.write(f"Final screenshots kept:\n")
            for kept_file in kept_files:
                log_file.write(f"  - {kept_file.name}\n")
        log_file.write(f"=" * 50 + "\n\n")
    
    return removed_count


def are_images_similar(image1_path, image2_path, threshold=None, top_ratio=None):
    """
    Compare two images and return the change percentage between them.
    Uses the same logic as detect_frame_change but for image files.
    
    Args:
        image1_path (str): Path to first image
        image2_path (str): Path to second image
        threshold (float): Threshold for considering change significant (default: Config.CHANGE_DETECTION_THRESHOLD = 4%)
        top_ratio (float): Ratio of top portion to analyze (default: Config.TOP_ANALYSIS_RATIO = 20%)
        
    Returns:
        float: Change percentage (0.0-1.0), or None if comparison failed
    """
    if threshold is None:
        threshold = Config.CHANGE_DETECTION_THRESHOLD
    if top_ratio is None:
        top_ratio = Config.TOP_ANALYSIS_RATIO
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
        
        # Threshold the difference (pixels with change > intensity threshold)
        _, thresh = cv2.threshold(diff, Config.PIXEL_INTENSITY_THRESHOLD, 255, cv2.THRESH_BINARY)
        
        # Calculate percentage of changed pixels
        total_pixels = top1.shape[0] * top1.shape[1]  # height * width
        changed_pixels = cv2.countNonZero(thresh)
        change_percentage = changed_pixels / total_pixels
        
        # Return the actual change percentage for logging
        return change_percentage
    except Exception as e:
        # If comparison fails for any reason, return None to indicate error
        return None


def _save_screenshot(frame, current_time, screenshot_count, screenshots_dir):
    """Save a screenshot with robust error handling and debug copy"""
    import shutil
    from pathlib import Path
    
    # Check if frame is valid
    if frame is None or frame.size == 0:
        return False
    
    # Create debug folder for A screenshots (without deleting existing contents)
    main_dir = Path(screenshots_dir).parent
    debug_a_dir = main_dir / "debug_a_screenshots"
    debug_a_dir.mkdir(exist_ok=True)  # Creates folder if it doesn't exist, but doesn't clear it
        
    # Create filename with timestamp
    timestamp_str = f"{int(current_time//60):02d}m{int(current_time%60):02d}s"
    filename = f"screenshot_{screenshot_count+1:03d}_{timestamp_str}.jpg"
    filepath = os.path.join(screenshots_dir, filename)
    debug_filepath = debug_a_dir / filename
    
    # Save the frame to main screenshots folder
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
    
    # If main save was successful, also save a debug copy
    if os.path.exists(filepath):
        try:
            # Copy the successfully saved file to debug folder
            shutil.copy2(filepath, debug_filepath)
        except Exception as e:
            # If debug copy fails, don't fail the whole operation
            pass
    
    # Check if file was actually created
    return os.path.exists(filepath)

def _save_screenshot_b(frame, current_time, a_screenshot_num, screenshots_dir):
    """Save a B screenshot with robust error handling"""
    # Check if frame is valid
    if frame is None or frame.size == 0:
        return False
        
    # Create filename with timestamp for B screenshot
    timestamp_str = f"{int(current_time//60):02d}m{int(current_time%60):02d}s"
    filename = f"screenshot_{a_screenshot_num:03d}_B_{timestamp_str}.jpg"
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
                    temp_filename = f"temp_screenshot_B_{a_screenshot_num}.jpg"
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

def _merge_b_to_a_screenshots(screenshots_dir, log_path):
    """
    Merge the left 20% of each B screenshot to the corresponding A screenshot and delete B screenshots.
    """
    from pathlib import Path
    import numpy as np
    
    # Get all A and B screenshot files
    screenshot_files = list(Path(screenshots_dir).glob("screenshot_*.jpg"))
    a_files = [f for f in screenshot_files if "_B_" not in f.name]
    b_files = [f for f in screenshot_files if "_B_" in f.name]
    
    # Sort A files by number
    a_files = sorted(a_files, key=lambda x: int(x.name.split('_')[1]))
    
    merged_count = 0
    deleted_count = 0
    
    with open(log_path, 'a', encoding='utf-8') as log_file:
        log_file.write(f"\n--- MERGING B SCREENSHOTS ---\n")
        log_file.write(f"A screenshots available for merging: {len(a_files)}\n")
        log_file.write(f"B screenshots available for merging: {len(b_files)}\n")
        for a_file in a_files:
            log_file.write(f"  A file: {a_file.name}\n")
        for b_file in b_files:
            log_file.write(f"  B file: {b_file.name}\n")
        log_file.write(f"\n")
    
    for a_file in a_files:
        # Extract the screenshot number from A file
        a_num = int(a_file.name.split('_')[1])
        
        # Find corresponding B file
        b_file = None
        for bf in b_files:
            if bf.name.startswith(f"screenshot_{a_num:03d}_B_"):
                b_file = bf
                break
        
        with open(log_path, 'a', encoding='utf-8') as log_file:
            if b_file:
                log_file.write(f"Found B file for A screenshot {a_num}: {b_file.name}\n")
            else:
                log_file.write(f"No B file found for A screenshot {a_num}\n")
        
        if b_file and b_file.exists():
            try:
                # Load both images
                img_a = cv2.imread(str(a_file))
                img_b = cv2.imread(str(b_file))
                
                if img_a is not None and img_b is not None:
                    # Get dimensions
                    height_a, width_a = img_a.shape[:2]
                    height_b, width_b = img_b.shape[:2]
                    
                    # Resize B to match A if needed
                    if height_b != height_a or width_b != width_a:
                        img_b = cv2.resize(img_b, (width_a, height_a))
                    
                    # Calculate 20% width
                    left_width = int(width_a * 0.2)
                    
                    # Extract left 20% from B
                    left_b = img_b[:, :left_width]
                    
                    # Replace left 20% of A with left 20% of B
                    img_a[:, :left_width] = left_b
                    
                    # Save the merged image back to A file
                    success = cv2.imwrite(str(a_file), img_a)
                    
                    if success:
                        merged_count += 1
                        with open(log_path, 'a', encoding='utf-8') as log_file:
                            log_file.write(f"✓ Merged {b_file.name} into {a_file.name}\n")
                    else:
                        with open(log_path, 'a', encoding='utf-8') as log_file:
                            log_file.write(f"✗ Failed to save merged image for {a_file.name}\n")
                
                # Delete the B file
                b_file.unlink()
                deleted_count += 1
                with open(log_path, 'a', encoding='utf-8') as log_file:
                    log_file.write(f"✓ Deleted {b_file.name}\n")
                    
            except Exception as e:
                with open(log_path, 'a', encoding='utf-8') as log_file:
                    log_file.write(f"✗ Error processing {a_file.name} and {b_file.name}: {e}\n")
        else:
            with open(log_path, 'a', encoding='utf-8') as log_file:
                log_file.write(f"- No B screenshot found for {a_file.name}\n")
    
    with open(log_path, 'a', encoding='utf-8') as log_file:
        log_file.write(f"--- MERGE COMPLETE ---\n")
        log_file.write(f"Merged: {merged_count} screenshots\n")
        log_file.write(f"Deleted: {deleted_count} B screenshots\n")
    
    print(f"Merged {merged_count} screenshots and deleted {deleted_count} B screenshots")

def main():
    parser = argparse.ArgumentParser(description="Extract screenshots from video files at regular intervals or based on content changes")
    parser.add_argument("video_path", help="Path to the video file")
    parser.add_argument("--start", type=float, default=2, help="Starting time in seconds (default: 2)")
    parser.add_argument("--interval", type=float, default=12, help="Interval between screenshots in seconds for time-based method (default: 12)")
    parser.add_argument("--method", choices=['time', 'change'], default='change', help="Screenshot method: 'time' for fixed intervals, 'change' for content change detection (default: change)")
    parser.add_argument("--change-threshold", type=float, default=Config.CHANGE_DETECTION_THRESHOLD, help=f"Threshold for change detection method (0.0-1.0, default: {Config.CHANGE_DETECTION_THRESHOLD} = {Config.CHANGE_DETECTION_THRESHOLD*100} percent)")
    parser.add_argument("--test", action="store_true", help="Test mode: only check video properties without extracting screenshots")
    parser.add_argument("--create-pdf", action="store_true", help="Create PDF after extracting screenshots")
    parser.add_argument("--custom-title", type=str, help="Custom title for PDF filename and content (supports Japanese)")
    parser.add_argument("--crop-ratio", type=float, default=0.32, help="Portion of height to keep from top for PDF (default: 0.32)")
    parser.add_argument("--strips-per-page", type=int, default=6, help="Maximum strips per A4 page (default: 6)")
    parser.add_argument("--recapture", action="store_true", help="Force recapture of screenshots, replacing existing raw folder")
    parser.add_argument("--disable-ocr", action="store_true", help="Disable OCR testing in duplicate detection (faster processing)")
    parser.add_argument("--disable-duplicate-detection", action="store_true", help="Disable all duplicate detection (fastest processing, keeps all screenshots)")
    
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
    # Determine folder name for main folder
    custom_folder_name = args.custom_title if args.custom_title else None
    
    result = extract_screenshots(
        args.video_path, 
        args.start, 
        args.interval, 
        args.method, 
        args.change_threshold,
        args.recapture,
        custom_folder_name,
        args.disable_ocr,
        args.disable_duplicate_detection
    )
    
    if isinstance(result, tuple) and len(result) == 3:
        success, ascii_folder_name, display_folder_name = result
    elif isinstance(result, tuple):
        success, screenshots_dir = result
        # Fallback for old return format
        video_name = Path(args.video_path).stem
        ascii_folder_name = sanitize_filename(video_name)
        display_folder_name = ascii_folder_name
    else:
        success = result
        # Fallback to old structure if needed
        video_name = Path(args.video_path).stem
        ascii_folder_name = sanitize_filename(video_name)
        display_folder_name = ascii_folder_name
    
    if not success:
        sys.exit(1)
    
    # Create PDF if requested
    if args.create_pdf:
        print("\n=== CREATING PDF ===")
        video_name = Path(args.video_path).stem
        sanitized_name = sanitize_filename(video_name)
        
        # Use ASCII folder name for file operations (to avoid Unicode path issues)
        scores_dir = os.path.join(os.path.dirname(os.getcwd()), "scores")
        main_folder = os.path.join(scores_dir, ascii_folder_name)
        
        # Prefer result folder (merged screenshots) if it exists, otherwise use main screenshots folder
        result_dir = os.path.join(main_folder, "screenshots", "result")
        screenshots_dir = os.path.join(main_folder, "screenshots")
        
        if os.path.exists(result_dir) and len(os.listdir(result_dir)) > 0:
            pdf_source_dir = result_dir
            print(f"Using merged screenshots from: {result_dir}")
        elif os.path.exists(screenshots_dir):
            pdf_source_dir = screenshots_dir
            print(f"Using original screenshots from: {screenshots_dir}")
        else:
            print(f"✗ No screenshots found in {screenshots_dir}")
            return
        
        if os.path.exists(pdf_source_dir):
            # Import create_pdf function directly instead of subprocess
            try:
                # Import the create_pdf module functions
                import importlib.util
                code_dir = os.path.dirname(os.path.abspath(__file__))
                create_pdf_path = os.path.join(code_dir, "create_pdf.py")
                spec = importlib.util.spec_from_file_location("create_pdf", create_pdf_path)
                if spec is not None and spec.loader is not None:
                    create_pdf_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(create_pdf_module)
                    
                    # Use custom title if provided, otherwise use sanitized video name
                    pdf_title = args.custom_title if args.custom_title else sanitized_name
                    pdf_title_safe = sanitize_filename(pdf_title) if args.custom_title else sanitized_name
                    
                    pdf_filename = f"{pdf_title_safe}_score.pdf"
                    pdf_path = os.path.join(main_folder, pdf_filename)
                    
                    print(f"Creating PDF: {pdf_path}")
                    success = create_pdf_module.create_pdf_from_screenshots(
                        pdf_source_dir,  # Use the selected source directory
                        pdf_path, 
                        args.crop_ratio, 
                        args.strips_per_page,
                        pdf_title  # Pass original title (with Japanese chars)
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
                    code_dir = os.path.dirname(os.path.abspath(__file__))
                    create_pdf_path = os.path.join(code_dir, "create_pdf.py")
                    cmd = [
                        sys.executable, create_pdf_path, pdf_source_dir, 
                        "--output", pdf_filename,
                        "--crop-ratio", str(args.crop_ratio),
                        "--strips-per-page", str(args.strips_per_page)
                    ]
                    
                    result = subprocess.run(
                        cmd, 
                        capture_output=True, 
                        text=True, 
                        cwd=os.path.dirname(os.path.abspath(__file__)),
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