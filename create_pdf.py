import cv2
import os
import numpy as np
from PIL import Image, ImageDraw
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfutils
from reportlab.lib.units import inch
import io
import argparse
from pathlib import Path

def crop_top_portion(image, ratio=0.32):
    """
    Crop the top portion of an image.
    
    Args:
        image: OpenCV image (numpy array)
        ratio: Portion of height to keep from top (default: 0.32 = 32%)
    
    Returns:
        Cropped image
    """
    height, width = image.shape[:2]
    crop_height = int(height * ratio)
    return image[0:crop_height, 0:width]

def stack_images_vertically(images):
    """
    Stack multiple images vertically.
    
    Args:
        images: List of OpenCV images
    
    Returns:
        Stacked image
    """
    if not images:
        return None
    
    # Find the maximum width to ensure consistent stacking
    max_width = max(img.shape[1] for img in images)
    
    # Resize all images to have the same width
    resized_images = []
    for img in images:
        if img.shape[1] != max_width:
            height = int(img.shape[0] * max_width / img.shape[1])
            resized_img = cv2.resize(img, (max_width, height))
            resized_images.append(resized_img)
        else:
            resized_images.append(img)
    
    # Stack vertically
    stacked = np.vstack(resized_images)
    return stacked

def fit_image_to_a4(image, a4_width, a4_height, margin=60, title_space=0):
    """
    Resize image to fit A4 page with margins and top alignment.
    
    Args:
        image: OpenCV image
        a4_width: A4 width in pixels
        a4_height: A4 height in pixels
        margin: Margin in pixels (increased default for better borders)
        title_space: Extra space reserved for title (pixels)
    
    Returns:
        Resized image that fits A4
    """
    available_width = a4_width - 2 * margin
    available_height = a4_height - margin - title_space  # Only subtract margin from top, strips align to top
    
    height, width = image.shape[:2]
    
    # Calculate scale to fit within available space
    scale_width = available_width / width
    scale_height = available_height / height
    scale = min(scale_width, scale_height)
    
    # Resize image
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    resized_image = cv2.resize(image, (new_width, new_height))
    return resized_image

def create_pdf_from_screenshots(screenshots_dir, output_pdf="stacked_screenshots.pdf", crop_ratio=0.32, strips_per_page=6, song_title=None):
    """
    Create PDF with stacked screenshot strips.
    
    Args:
        screenshots_dir: Directory containing screenshots
        output_pdf: Output PDF filename
        crop_ratio: Portion of height to keep from top
        strips_per_page: Maximum number of strips per A4 page
        song_title: Title to display on first page
    """
    # Get all screenshot files
    screenshot_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        screenshot_files.extend(Path(screenshots_dir).glob(ext))
    
    if not screenshot_files:
        print(f"No screenshot files found in {screenshots_dir}")
        return False
    
    # Sort files by name to maintain order
    screenshot_files.sort()
    print(f"Found {len(screenshot_files)} screenshots")
    
    # A4 dimensions in pixels (at 300 DPI)
    a4_width = int(8.27 * 300)  # 2481 pixels
    a4_height = int(11.69 * 300)  # 3507 pixels
    
    # Create PDF
    pdf_path = os.path.join(os.getcwd(), output_pdf)
    c = canvas.Canvas(pdf_path, pagesize=A4)
    
    cropped_strips = []
    page_count = 0
    
    print(f"Processing screenshots...")
    
    for i, screenshot_file in enumerate(screenshot_files):
        print(f"Processing {screenshot_file.name}...")
        
        # Load image
        image = cv2.imread(str(screenshot_file))
        if image is None:
            print(f"Warning: Could not load {screenshot_file}")
            continue
        
        # Crop top portion
        cropped = crop_top_portion(image, crop_ratio)
        cropped_strips.append(cropped)
        
        # When we have enough strips or reached the end, create a page
        if len(cropped_strips) >= strips_per_page or i == len(screenshot_files) - 1:
            # Stack all strips vertically
            stacked_image = stack_images_vertically(cropped_strips)
            
            if stacked_image is not None:
                page_count += 1
                is_first_page = (page_count == 1)
                title_space_pixels = 60 if (is_first_page and song_title) else 0  # Reserve space for title
                
                # Fit to A4 size (account for title space on first page)
                fitted_image = fit_image_to_a4(stacked_image, a4_width, a4_height, title_space=title_space_pixels)
                
                # Convert to PIL Image for PDF
                fitted_rgb = cv2.cvtColor(fitted_image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(fitted_rgb)
                
                # Create image buffer
                img_buffer = io.BytesIO()
                pil_image.save(img_buffer, format='JPEG', quality=95)
                img_buffer.seek(0)
                
                print(f"Creating page {page_count} with {len(cropped_strips)} strips")
                
                # Add title on first page
                if is_first_page and song_title:
                    # Set font and size for title
                    c.setFont("Helvetica-Bold", 24)
                    
                    # Convert title to title case for display (capitalize first letter of each word)
                    display_title = song_title.title()
                    
                    # Calculate title position (centered horizontally, with consistent top margin)
                    title_width = c.stringWidth(display_title, "Helvetica-Bold", 24)
                    title_x = (A4[0] - title_width) / 2
                    title_y = A4[1] - 40  # 40 points from top (about 0.56 inches) for title
                    
                    # Draw title
                    c.drawString(title_x, title_y, display_title)
                
                # Calculate position to align to top with margin (adjust for title on first page)
                img_width, img_height = fitted_image.shape[1], fitted_image.shape[0]
                x = (A4[0] - img_width * 72/300) / 2  # Center horizontally, convert pixels to points
                
                # Set top margin (border from top edge)
                top_margin = 60  # 60 points from top (about 0.83 inches)
                
                if is_first_page and song_title:
                    # Position image below title with minimal padding
                    y = A4[1] - 60 - (img_height * 72/300)  # 60 points from top to account for title + margin
                else:
                    # Align to top with margin
                    y = A4[1] - top_margin - (img_height * 72/300)
                
                c.drawImage(ImageReader(img_buffer), x, y, 
                           width=img_width * 72/300, height=img_height * 72/300)
                
                c.showPage()
            
            # Reset for next page
            cropped_strips = []
    
    # Save PDF
    c.save()
    print(f"\nPDF created successfully: {pdf_path}")
    print(f"Total pages: {page_count}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Create PDF from screenshot strips")
    parser.add_argument("screenshots_dir", help="Directory containing screenshots")
    parser.add_argument("--output", default="stacked_screenshots.pdf", help="Output PDF filename")
    parser.add_argument("--crop-ratio", type=float, default=0.32, help="Portion of height to keep from top (default: 0.32)")
    parser.add_argument("--strips-per-page", type=int, default=6, help="Maximum strips per A4 page (default: 6)")
    parser.add_argument("--title", help="Song title to display on first page")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.screenshots_dir):
        print(f"Error: Directory '{args.screenshots_dir}' does not exist.")
        return
    
    success = create_pdf_from_screenshots(
        args.screenshots_dir, 
        args.output, 
        args.crop_ratio, 
        args.strips_per_page,
        args.title
    )
    
    if not success:
        print("Failed to create PDF")

if __name__ == "__main__":
    main()
