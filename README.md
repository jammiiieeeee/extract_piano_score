# Video Screenshot Extractor & PDF Creator

This project extracts screenshots from video files at regular intervals and optionally creates PDF documents with cropped and stacked screenshots.

## Requirements

- Python 3.x
- OpenCV (cv2)
- Pillow (PIL)
- ReportLab

## Supported Video Formats

- MP4
- AVI  
- MOV
- MKV
- WebM

## Installation

The required packages are already installed:
```bash
pip install opencv-python Pillow reportlab
```

## Usage

### Basic Usage - Extract Screenshots Only

Extract screenshots at 2, 14, 26, 38... seconds:
```bash
python main.py "path/to/your/video.mp4"
# or
python main.py "path/to/your/video.webm"
```

### Extract Screenshots AND Create PDF

Extract screenshots and automatically create a PDF with stacked strips:
```bash
python main.py "path/to/your/video.mp4" --create-pdf
```

### Custom Parameters

```bash
python main.py "path/to/your/video.mp4" --start 5 --interval 10 --create-pdf --crop-ratio 0.4 --strips-per-page 6
```

### Create PDF from Existing Screenshots

If you already have screenshots, you can create a PDF directly:
```bash
python create_pdf.py "screenshot_folder_name"
```

## Parameters

### Screenshot Extraction (main.py):
- `video_path`: Path to the video file (required)
- `--start`: Starting time in seconds (default: 2)
- `--interval`: Interval between screenshots in seconds (default: 12)
- `--test`: Test mode - check video properties without extracting
- `--create-pdf`: Create PDF after extracting screenshots
- `--crop-ratio`: Portion of height to keep from top for PDF (default: 0.32 = 32%)
- `--strips-per-page`: Maximum strips per A4 page (default: 5)

### PDF Creation (create_pdf.py):
- `screenshots_dir`: Directory containing screenshots (required)
- `--output`: Output PDF filename (default: "stacked_screenshots.pdf")
- `--crop-ratio`: Portion of height to keep from top (default: 0.32)
- `--strips-per-page`: Maximum strips per A4 page (default: 5)

## Output

### Screenshots
- Creates a folder named `{video_name}_screenshots` in the current directory
- Screenshots are saved as JPEG files with timestamps in the filename
- Format: `screenshot_001_00m02s.jpg`, `screenshot_002_00m14s.jpg`, etc.

### PDF
- Creates a PDF file named `{video_name}_stacked.pdf`
- Each page contains up to 5 screenshot strips (configurable)
- Each strip shows the top 32% of the original screenshot (configurable)
- Strips are stacked vertically, with later screenshots below earlier ones
- Automatically fits to A4 page size with margins

## Examples

### Example 1: Piano Tutorial Video
```bash
python main.py "C:\Videos\piano_lesson.mp4" --create-pdf
```

This will:
1. Extract screenshots at 2, 14, 26, 38... seconds
2. Save them in `piano_lesson_screenshots/`
3. Create `piano_lesson_stacked.pdf` with cropped strips

### Example 2: Custom Settings for Music Sheets
```bash
python main.py "music_video.mp4" --start 3 --interval 15 --create-pdf --crop-ratio 0.4 --strips-per-page 4
```

This will:
1. Start at 3 seconds, take screenshots every 15 seconds
2. Crop top 40% of each screenshot
3. Put 4 strips per PDF page

### Example 3: Process Existing Screenshots
```bash
python create_pdf.py "existing_screenshots_folder" --output "my_music_sheets.pdf" --crop-ratio 0.35
```

## Features

- **Handles Japanese/Unicode characters** in filenames
- **Multiple fallback save methods** for problematic file paths
- **Automatic filename sanitization** 
- **Detailed progress reporting** and error handling
- **Flexible PDF layout** with customizable cropping and page layout
- **A4-optimized output** with proper margins and scaling
