# Video Screenshot Extractor - GUI Application

A user-friendly graphical interface for extracting screenshots from videos and creating PDF sheet music.

## 🚀 Quick Start

### Windows Users
1. Double-click `Launch Video Extractor GUI.bat`
2. Or run `python launch_gui.py`

### Mac/Linux Users
```bash
python3 launch_gui.py
```

## 📋 How to Use

### 1. Select Video File
- Click **Browse** to select a video file
- Or drag and drop a video file directly onto the application window
- Supported formats: .mp4, .avi, .mov, .mkv, .webm

### 2. Preview & Adjust Crop Ratio
- Click **Load Video Preview** to see the middle frame of your video
- Use the **Crop Ratio** slider to adjust how much of the top portion to keep
- Or click and drag the red line directly on the video preview
- The crop ratio determines what part of the video will be included in the final PDF

### 3. Configure Parameters

#### Extraction Settings
- **Start Time (s)**: When to start extracting (default: 2 seconds)
- **Interval (s)**: Time between screenshots for time-based method (default: 12 seconds)
- **Method**: 
  - *Time*: Extract at fixed intervals
  - *Change*: Extract when content changes (recommended)
- **Change Threshold**: How much change needed to trigger a new screenshot (0.0-1.0)

#### PDF Settings
- **Strips per Page**: How many screenshot strips per PDF page (default: 7)
- **Custom Title**: Custom name for the PDF file and title
- **Crop Ratio**: Portion of video height to keep from the top (adjustable with slider)

#### Options (Checkboxes)
- ✅ **Create PDF**: Generate PDF after extraction (recommended)
- ☐ **Test Mode Only**: Just check video properties without extracting
- ☐ **Skip PDF Title**: Don't show title on PDF pages
- ☐ **Force Recapture**: Re-extract even if screenshots already exist
- ☐ **Disable OCR**: Faster processing, skip text detection
- ☐ **Disable Duplicate Detection**: Fastest processing, keep all screenshots

### 4. Start Extraction
- Click **🎬 Start Extraction** to begin the process
- Monitor progress in the progress bar and status text
- The application will show when extraction is complete

### 5. View Results
- If PDF creation is enabled, you'll be asked if you want to open the PDF automatically
- Screenshots and logs are saved in the `scores/[video_name]/` folder

## 📁 Output Structure
```
scores/
└── [video_name]/
    ├── screenshots/
    │   ├── raw/           # Original A and B screenshots
    │   ├── result/        # Merged unique screenshots
    │   └── duplicates/    # Detected duplicate pairs
    ├── [video_name]_score.pdf   # Generated PDF
    ├── [video_name]_log.txt     # Processing log
    └── [video_name]_similarity_heatmap.html  # Analysis dashboard
```

## 🎼 Features

- **Smart Change Detection**: Automatically detects when sheet music changes
- **Duplicate Removal**: Advanced algorithms to remove duplicate screenshots
- **Visual Crop Adjustment**: See exactly what will be included in your PDF
- **Progress Visualization**: Beautiful analysis dashboards and heatmaps
- **Japanese Support**: Custom titles with Japanese characters supported
- **Drag & Drop**: Simply drag video files onto the application

## 🔧 Requirements

- Python 3.7+
- OpenCV
- Tkinter (usually included with Python)
- Pillow (PIL)
- ReportLab
- PaddleOCR (for text detection)

Install requirements:
```bash
pip install -r requirements.txt
```

## 💡 Tips

1. **For Piano Videos**: Use change detection method with default threshold (0.04)
2. **For Fast Music**: Lower the change threshold to catch quick changes
3. **For Slow Music**: Increase the change threshold to avoid too many screenshots
4. **PDF Quality**: Adjust crop ratio to include only the sheet music, excluding hands/piano
5. **Performance**: Disable OCR and duplicate detection for fastest processing

## 🐛 Troubleshooting

**GUI won't start?**
- Make sure Python and tkinter are installed
- Try: `python -m tkinter` to test tkinter installation

**Video won't load?**
- Check if the video file is corrupted
- Try converting to a different format (MP4 recommended)

**PDF creation fails?**
- Make sure ReportLab is installed: `pip install reportlab`
- Check if output directory is writable

**Drag & drop not working?**
- This feature requires `tkinterdnd2` package
- Install with: `pip install tkinterdnd2`

## 🎨 Advanced Usage

For batch processing or automation, you can still use the command-line interface:
```bash
python code/main.py path/to/video.mp4 --create-pdf --crop-ratio 0.32
```

See `python code/main.py --help` for all command-line options.