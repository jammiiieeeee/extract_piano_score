# Piano Score Extractor

Automatically extract and convert piano tutorials from video to organized, printable PDF scores.

## Setup Guide

### Prerequisites
- Python 3.x

### Installation
1. Clone or download this repository
2. Install the required packages:
```bash
pip install opencv-python Pillow reportlab
```

### Basic Usage
To extract screenshots and create a PDF from a video:
```bash
python main.py "path/to/your/video.mp4" --create-pdf
```

This will:
1. Extract screenshots at regular intervals from your video
2. Create a PDF with the extracted content organized in a printable format
3. Save the output in the same directory

## Operand Usages

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

### Examples:
1. Basic extraction with PDF creation:
```bash
python main.py "piano_tutorial.mp4" --create-pdf
```

2. Custom settings for music sheets:
```bash
python main.py "music_video.mp4" --start 3 --interval 15 --create-pdf --crop-ratio 0.4 --strips-per-page 4
```

3. Create PDF from existing screenshots:
```bash
python create_pdf.py "existing_screenshots_folder" --output "my_music_sheets.pdf" --crop-ratio 0.35
```

### Features:
- Handles Japanese/Unicode characters in filenames
- Multiple fallback save methods for problematic file paths
- Automatic filename sanitization 
- Detailed progress reporting and error handling
- Flexible PDF layout with customizable cropping and page layout
- A4-optimized output with proper margins and scaling