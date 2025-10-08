# Piano Score Extractor

Automatically extract and convert piano tutorials from video to organized, printable PDF scores using intelligent content detection or fixed time intervals.

## Features

- **Two extraction methods**: Time-based intervals or intelligent content change detection
- **Smart scene change detection**: Automatically stops extraction when major scene changes occur (>70% change)
- **Unicode/Japanese filename support**: Handles complex characters in video filenames
- **Organized output structure**: Creates main folders with screenshots and PDFs organized together
- **Detailed logging**: Comprehensive analysis logs with change percentages and timestamps
- **Progress tracking**: Real-time progress bars with ETA estimates
- **PDF generation**: Automatically creates cropped and stacked sheet music PDFs

## Setup Guide

### Prerequisites
- Python 3.x

### Installation
1. Clone or download this repository
2. Install the required packages:
```bash
pip install opencv-python Pillow reportlab
```

## Usage

### Basic Usage
Extract screenshots and create a PDF from a video:
```bash
python main.py "path/to/your/video.mp4" --create-pdf
```

### Advanced Usage with Change Detection
Use intelligent content change detection (recommended for piano tutorials):
```bash
python main.py "path/to/your/video.mp4" --method change --create-pdf
```

### Test Video Properties
Check if a video can be processed without extracting screenshots:
```bash
python main.py "path/to/your/video.mp4" --test
```

## Parameters

### Screenshot Extraction (main.py):
- `video_path`: Path to the video file (required)
- `--start`: Starting time in seconds (default: 2)
- `--interval`: Interval between screenshots in seconds for time-based method (default: 12)
- `--method`: Screenshot method - 'time' for fixed intervals, 'change' for content detection (default: time)
- `--change-threshold`: Threshold for change detection method (0.0-1.0, default: 0.04 = 4%)
- `--test`: Test mode - check video properties without extracting
- `--create-pdf`: Create PDF after extracting screenshots
- `--crop-ratio`: Portion of height to keep from top for PDF (default: 0.32 = 32%)
- `--strips-per-page`: Maximum strips per A4 page (default: 6)

### PDF Creation (create_pdf.py):
- `screenshots_dir`: Directory containing screenshots (required)
- `--output`: Output PDF filename (default: "stacked_screenshots.pdf")
- `--crop-ratio`: Portion of height to keep from top (default: 0.32)
- `--strips-per-page`: Maximum strips per A4 page (default: 6)

## Output Structure

The tool creates an organized folder structure:
```
[video_name]/
├── screenshots/
│   ├── screenshot_001_00m02s.jpg
│   ├── screenshot_002_00m14s.jpg
│   └── ...
├── [video_name]_score.pdf
└── [video_name]_log.txt
```

### Screenshots
- Saved as JPEG files with timestamps in the filename
- Format: `screenshot_001_00m02s.jpg`, `screenshot_002_00m14s.jpg`, etc.

### PDF
- Named `{video_name}_score.pdf`
- Each page contains up to 6 screenshot strips (configurable)
- Each strip shows the top 32% of the original screenshot (configurable)
- Strips are stacked vertically with later screenshots below earlier ones
- Automatically fits to A4 page size with margins

### Log File
- Named `{video_name}_log.txt`
- Contains detailed analysis information including:
  - Video properties and settings used
  - Timestamp and change percentage for each frame analyzed
  - Screenshot capture events with reasons
  - Major scene change detection events

## Examples

### Example 1: Basic Piano Tutorial Processing
```bash
python main.py "C:\Videos\piano_lesson.mp4" --create-pdf
```
This will:
1. Extract screenshots at 2, 14, 26, 38... seconds using time-based method
2. Save them in `piano_lesson/screenshots/`
3. Create `piano_lesson/piano_lesson_score.pdf` with cropped strips
4. Generate `piano_lesson/piano_lesson_log.txt` with analysis details

### Example 2: Intelligent Change Detection (Recommended)
```bash
python main.py "music_video.mp4" --method change --create-pdf --change-threshold 0.06
```
This will:
1. Analyze the video for content changes using 6% threshold
2. Automatically capture screenshots when significant changes are detected
3. Stop processing if major scene changes (>70%) are detected
4. Create organized output with PDF and detailed logs

### Example 3: Custom Settings for Music Sheets
```bash
python main.py "music_video.mp4" --start 3 --interval 15 --create-pdf --crop-ratio 0.4 --strips-per-page 4
```
This will:
1. Start at 3 seconds, take screenshots every 15 seconds
2. Crop top 40% of each screenshot
3. Put 4 strips per PDF page

### Example 4: Test Video Before Processing
```bash
python main.py "path/to/video.mp4" --test
```
This will display video properties without extracting any screenshots.

### Example 5: Create PDF from Existing Screenshots
```bash
python create_pdf.py "existing_screenshots_folder" --output "my_music_sheets.pdf" --crop-ratio 0.35
```

## Detection Methods

### Time-based Method (--method time)
- Takes screenshots at fixed intervals (default: every 12 seconds)
- Predictable output with consistent spacing
- Good for videos with regular content changes
- Faster processing

### Change Detection Method (--method change)
- Analyzes top 20% of each frame for content changes
- Captures screenshots only when significant changes occur (default: 4% threshold)
- Automatically stops when major scene changes detected (>70% change)
- Ideal for piano tutorials where sheet music changes
- More intelligent but slower processing

## Advanced Features

### Smart Scene Change Detection
- Monitors for major changes (>70%) that indicate scene transitions
- Automatically aborts screenshot capture to avoid capturing non-music content
- Prevents contamination of sheet music with unrelated video content

### Unicode/International Character Support
- Handles Japanese, Chinese, and other Unicode characters in filenames
- Automatic filename sanitization for filesystem compatibility
- Multiple fallback methods for file saving

### Robust Error Handling
- Multiple fallback methods for screenshot saving
- Comprehensive error logging
- Graceful handling of corrupted or problematic video files

## Supported Video Formats

- MP4 (recommended)
- AVI
- MOV
- MKV
- WebM

## Troubleshooting

### Common Issues

**Video won't open:**
- Use `--test` flag to check if the video file is readable
- Ensure the video format is supported
- Check if the file path contains special characters

**No screenshots captured with change detection:**
- Try lowering the `--change-threshold` (e.g., 0.02 for 2%)
- Switch to time-based method if content changes are minimal
- Check the log file for change percentages

**PDF creation fails:**
- Ensure screenshots were successfully extracted first
- Check that the screenshots directory exists and contains images
- Verify sufficient disk space for PDF generation

**Progress stops early:**
- This may indicate major scene change detection (>70% change)
- Check the log file for scene change events
- Consider using time-based method if this is unwanted behavior

### Performance Tips

- Use change detection method for piano tutorials and sheet music videos
- Use time-based method for consistent content or faster processing
- Adjust `--change-threshold` based on your video content:
  - Piano tutorials: 0.04-0.08 (4-8%)
  - Guitar tabs: 0.03-0.06 (3-6%)
  - General music videos: 0.02-0.04 (2-4%)

## Contributing

Feel free to submit issues and enhancement requests!