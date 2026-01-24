# OCR Enhancement Summary

## Overview
Enhanced the piano score extraction system with OCR (Optical Character Recognition) functionality to improve duplicate detection accuracy.

## New Dependencies
- **pytesseract**: Python wrapper for Google's Tesseract OCR engine
- **Tesseract OCR**: The actual OCR engine (installed via winget)

## Key Features Added

### 1. OCR Number Extraction
- **Function**: `extract_ocr_numbers(image_path)`
- **Purpose**: Extracts numbers from the top `DEFAULT_CROP_RATIO` (32%) portion of screenshots
- **Configuration**: Uses `Config.OCR_CROP_RATIO` which defaults to `Config.DEFAULT_CROP_RATIO` (0.32)
- **OCR Settings**: Configured to extract only numbers and decimal points
- **Return**: List of floating-point numbers found in the image

### 2. Enhanced Duplicate Detection
The duplicate detection process now includes a third test:

#### Previous Tests:
1. **Test 1**: Pixel similarity (≥95% threshold)
2. **Test 2**: Row similarity (≥94% coverage with ≥98% per-row threshold)

#### New Test:
3. **Test 3**: OCR number comparison
   - Extracts numbers from both images
   - Compares the extracted numbers
   - If numbers are different → **NOT a duplicate** (even if other tests pass)
   - If numbers are same → Proceed with existing duplicate logic

### 3. Duplicate Logic Enhancement
```
OLD LOGIC: Duplicate = (Test 1 OR Test 2) passes
NEW LOGIC: Duplicate = (Test 1 OR Test 2) passes AND (OCR numbers are same)
```

This prevents false positives where images look similar but represent different measures/bars of music.

## Configuration Parameters

### New Config Options:
```python
OCR_CROP_RATIO = DEFAULT_CROP_RATIO     # Use same ratio as PDF crop (32%)
OCR_CONFIDENCE_THRESHOLD = 30           # Minimum confidence for OCR text extraction
```

## Installation Requirements

### Automatic (already done):
- `pytesseract` Python package ✅
- `Pillow` Python package ✅

### Manual (already installed):
- Tesseract OCR executable ✅
  - Installed via: `winget install UB-Mannheim.TesseractOCR`
  - Auto-detected at: `C:\Program Files\Tesseract-OCR\tesseract.exe`

## Enhanced Logging

The duplicate detection now logs additional information:
- OCR numbers extracted from each image
- Whether OCR test passed or failed
- Detailed comparison results

### Log Example:
```
DUPLICATE: 01_02m14s_A.jpg vs 03_02m38s_A.jpg
  Test 1 (pixel): PASS
  Test 2 (row): FAIL  
  Test 3 (OCR): SAME
  OCR Numbers 1: [23.0, 4.5]
  OCR Numbers 2: [23.0, 4.5]
  Saved as: duplicate_pair_001

NOT DUPLICATE (OCR DIFFERENT): 02_02m26s_A.jpg vs 04_02m50s_A.jpg
  Test 1 (pixel): PASS
  Test 2 (row): PASS
  Test 3 (OCR): DIFFERENT
  OCR Numbers 1: [23.0, 4.5]
  OCR Numbers 2: [24.0, 4.5]
```

## Benefits

1. **Higher Accuracy**: Reduces false positives in duplicate detection
2. **Music-Specific**: Tailored for piano scores where measure numbers are crucial
3. **Configurable**: OCR region matches PDF crop region for consistency
4. **Robust**: Handles OCR failures gracefully
5. **Detailed Logging**: Enhanced debugging and analysis capabilities

## Usage

No changes to command-line usage. The OCR enhancement works automatically:

```bash
# Time-based extraction with enhanced duplicate detection
python main.py video.mp4 --method time

# Change-based extraction with enhanced duplicate detection  
python main.py video.mp4 --method change
```

The OCR functionality activates automatically during the duplicate detection phase.