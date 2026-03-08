#!/usr/bin/env python3
"""
GUI Application for Video Screenshot Extraction
Author: Assistant
Description: User-friendly interface for extracting screenshots from videos and creating PDFs
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import os
import sys
import threading
import subprocess
from pathlib import Path
from PIL import Image, ImageTk
import time

# Add the current directory to sys.path to import main.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from main import extract_screenshots, Config
except ImportError as e:
    print(f"Error importing main module: {e}")
    sys.exit(1)

class VideoExtractorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Screenshot Extractor")
        self.root.geometry("950x800")
        self.root.configure(bg='#f0f0f0')
        
        # Variables
        self.video_path = tk.StringVar()
        self.video_cap = None
        self.current_frame = None
        self.total_duration = 0
        self.video_fps = 0
        
        # Parameter variables with defaults
        self.start_time = tk.DoubleVar(value=2.0)
        self.interval = tk.DoubleVar(value=12.0)
        self.method = tk.StringVar(value='change')
        self.change_threshold = tk.DoubleVar(value=Config.CHANGE_DETECTION_THRESHOLD)
        self.test_mode = tk.BooleanVar(value=False)
        self.create_pdf = tk.BooleanVar(value=True)  # Default True per user request
        self.custom_title = tk.StringVar()
        self.skip_pdf_title = tk.BooleanVar(value=False)
        self.crop_ratio = tk.DoubleVar(value=Config.DEFAULT_CROP_RATIO)
        self.strips_per_page = tk.IntVar(value=Config.DEFAULT_STRIPS_PER_PAGE)
        self.recapture = tk.BooleanVar(value=False)
        self.disable_ocr = tk.BooleanVar(value=False)
        self.disable_duplicate_detection = tk.BooleanVar(value=False)
        
        # UI State
        self.is_processing = False
        self.crop_line_id = None
        
        self.create_widgets()
        self.setup_drag_drop()
        
    def create_widgets(self):
        """Create all UI widgets"""
        # Create main frame with scrollbar
        canvas = tk.Canvas(self.root)
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        # Configure scrolling
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Main content frame
        main_frame = ttk.Frame(scrollable_frame, padding="10")
        main_frame.pack(fill="both", expand=True)
        
        # Configure grid weights
        main_frame.columnconfigure(1, weight=1)
        
        # Enable mouse wheel scrolling
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", on_mousewheel)
        
        row = 0
        
        # === VIDEO SELECTION ===
        ttk.Label(main_frame, text="Video File:", font=('Arial', 12, 'bold')).grid(
            row=row, column=0, columnspan=3, sticky=tk.W, pady=(0, 10))
        row += 1
        
        # Video path frame
        video_frame = ttk.Frame(main_frame)
        video_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        video_frame.columnconfigure(0, weight=1)
        
        self.video_entry = ttk.Entry(video_frame, textvariable=self.video_path, font=('Arial', 10))
        self.video_entry.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 10))
        
        ttk.Button(video_frame, text="Browse", command=self.browse_video).grid(
            row=0, column=1)
        
        # Drag and drop hint
        ttk.Label(main_frame, text="💡 Tip: You can also drag and drop a video file onto this window", 
                 foreground='gray').grid(row=row+1, column=0, columnspan=3, sticky=tk.W, pady=(0, 15))
        row += 2
        
        # === VIDEO PREVIEW ===
        self.preview_frame = ttk.LabelFrame(main_frame, text="Video Preview & Crop Adjustment", 
                                          padding="10")
        self.preview_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 15))
        self.preview_frame.columnconfigure(0, weight=1)
        
        # Video preview canvas
        self.canvas = tk.Canvas(self.preview_frame, width=640, height=360, bg='black')
        self.canvas.grid(row=0, column=0, columnspan=3, pady=(0, 10))
        
        # Crop ratio controls
        crop_controls = ttk.Frame(self.preview_frame)
        crop_controls.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E))
        crop_controls.columnconfigure(1, weight=1)
        
        ttk.Label(crop_controls, text="Crop Ratio:").grid(row=0, column=0, padx=(0, 5))
        
        self.crop_scale = ttk.Scale(crop_controls, from_=0.1, to=0.8, orient=tk.HORIZONTAL,
                                   variable=self.crop_ratio, command=self.update_crop_preview)
        self.crop_scale.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 5))
        
        self.crop_value_label = ttk.Label(crop_controls, text=f"{self.crop_ratio.get():.2f}")
        self.crop_value_label.grid(row=0, column=2)
        
        ttk.Button(self.preview_frame, text="Load Video Preview", 
                  command=self.load_video_preview).grid(row=2, column=1, pady=(10, 0))
        
        row += 1
        
        # === PARAMETERS ===
        params_frame = ttk.LabelFrame(main_frame, text="Extraction Parameters", padding="10")
        params_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 15))
        params_frame.columnconfigure(1, weight=1)
        params_frame.columnconfigure(3, weight=1)
        
        param_row = 0
        
        # Start time
        ttk.Label(params_frame, text="Start Time (s):").grid(row=param_row, column=0, sticky=tk.W, padx=(0, 5))
        ttk.Entry(params_frame, textvariable=self.start_time, width=10).grid(row=param_row, column=1, sticky=tk.W, padx=(0, 20))
        
        # Interval
        ttk.Label(params_frame, text="Interval (s):").grid(row=param_row, column=2, sticky=tk.W, padx=(0, 5))
        ttk.Entry(params_frame, textvariable=self.interval, width=10).grid(row=param_row, column=3, sticky=tk.W)
        param_row += 1
        
        # Method
        ttk.Label(params_frame, text="Method:").grid(row=param_row, column=0, sticky=tk.W, padx=(0, 5), pady=(5, 0))
        method_frame = ttk.Frame(params_frame)
        method_frame.grid(row=param_row, column=1, sticky=tk.W, padx=(0, 20), pady=(5, 0))
        ttk.Radiobutton(method_frame, text="Time", variable=self.method, value='time').grid(row=0, column=0)
        ttk.Radiobutton(method_frame, text="Change", variable=self.method, value='change').grid(row=0, column=1, padx=(10, 0))
        
        # Change threshold
        ttk.Label(params_frame, text="Change Threshold:").grid(row=param_row, column=2, sticky=tk.W, padx=(0, 5), pady=(5, 0))
        ttk.Entry(params_frame, textvariable=self.change_threshold, width=10).grid(row=param_row, column=3, sticky=tk.W, pady=(5, 0))
        param_row += 1
        
        # Strips per page
        ttk.Label(params_frame, text="Strips per Page:").grid(row=param_row, column=0, sticky=tk.W, padx=(0, 5), pady=(5, 0))
        ttk.Entry(params_frame, textvariable=self.strips_per_page, width=10).grid(row=param_row, column=1, sticky=tk.W, padx=(0, 20), pady=(5, 0))
        
        # Custom title
        ttk.Label(params_frame, text="Custom Title:").grid(row=param_row, column=2, sticky=tk.W, padx=(0, 5), pady=(5, 0))
        ttk.Entry(params_frame, textvariable=self.custom_title, width=20).grid(row=param_row, column=3, sticky=(tk.W, tk.E), pady=(5, 0))
        param_row += 1
        
        # === CHECKBOXES ===
        checkbox_frame = ttk.LabelFrame(main_frame, text="Options", padding="10")
        checkbox_frame.grid(row=row+1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 15))
        
        # Arrange checkboxes in 2 columns
        checkboxes = [
            ("Test Mode Only", self.test_mode),
            ("Create PDF", self.create_pdf),
            ("Skip PDF Title", self.skip_pdf_title),
            ("Force Recapture", self.recapture),
            ("Disable OCR", self.disable_ocr),
            ("Disable Duplicate Detection", self.disable_duplicate_detection),
        ]
        
        for i, (text, var) in enumerate(checkboxes):
            col = i % 2
            row_offset = i // 2
            # Use regular tk.Checkbutton for reliable checkmark display
            try:
                # Get the frame background color to match
                frame_bg = checkbox_frame.cget('bg')
            except:
                frame_bg = '#f0f0f0'  # Default light gray
                
            checkbox = tk.Checkbutton(checkbox_frame, text=text, variable=var, 
                                    bg=frame_bg, 
                                    activebackground=frame_bg,
                                    relief='flat', 
                                    highlightthickness=0,
                                    font=('Segoe UI', 9))
            checkbox.grid(row=row_offset, column=col, sticky=tk.W, padx=(0, 40), pady=2)
        
        row += 2
        
        # === PROCESS BUTTON ===
        self.process_button = ttk.Button(main_frame, text="🎬 Start Extraction", 
                                       command=self.start_extraction, style='Accent.TButton')
        self.process_button.grid(row=row, column=0, columnspan=3, pady=(0, 15))
        
        row += 1
        
        # === PROGRESS ===
        self.progress_frame = ttk.LabelFrame(main_frame, text="Progress", padding="10")
        self.progress_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E))
        self.progress_frame.columnconfigure(0, weight=1)
        
        self.progress_var = tk.StringVar(value="Ready to start extraction...")
        self.progress_label = ttk.Label(self.progress_frame, textvariable=self.progress_var)
        self.progress_label.grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        self.progress_bar = ttk.Progressbar(self.progress_frame, mode='indeterminate')
        self.progress_bar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        # Bind events
        self.video_path.trace('w', self.on_video_path_change)
        self.crop_ratio.trace('w', self.on_crop_ratio_change)
        
    def setup_drag_drop(self):
        """Setup drag and drop functionality for video files"""
        try:
            from tkinterdnd2 import DND_FILES, TkinterDnD
            self.root = TkinterDnD.Tk()
            self.root.drop_target_register(DND_FILES)
            self.root.dnd_bind('<<Drop>>', self.on_drop)
        except ImportError:
            # tkinterdnd2 not available, skip drag-drop functionality
            pass
    
    def on_drop(self, event):
        """Handle drag and drop events"""
        files = self.root.tk.splitlist(event.data)
        if files:
            video_file = files[0]
            if any(video_file.lower().endswith(ext) for ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']):
                self.video_path.set(video_file)
            else:
                messagebox.showwarning("Invalid File", "Please drop a valid video file (.mp4, .avi, .mov, .mkv, .webm)")
    
    def browse_video(self):
        """Open file browser for video selection"""
        filetypes = [
            ("Video files", "*.mp4 *.avi *.mov *.mkv *.webm"),
            ("All files", "*.*")
        ]
        filename = filedialog.askopenfilename(title="Select Video File", filetypes=filetypes)
        if filename:
            self.video_path.set(filename)
    
    def on_video_path_change(self, *args):
        """Called when video path changes"""
        path = self.video_path.get()
        if path and os.path.exists(path):
            self.load_video_preview()
    
    def load_video_preview(self):
        """Load video and display middle frame"""
        video_path = self.video_path.get()
        if not video_path or not os.path.exists(video_path):
            messagebox.showerror("Error", "Please select a valid video file first")
            return
        
        try:
            # Release previous video if any
            if self.video_cap:
                self.video_cap.release()
            
            # Open video
            self.video_cap = cv2.VideoCapture(video_path)
            if not self.video_cap.isOpened():
                messagebox.showerror("Error", "Could not open video file")
                return
            
            # Get video properties
            self.video_fps = self.video_cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.total_duration = total_frames / self.video_fps if self.video_fps > 0 else 0
            width = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Go to middle frame
            middle_frame = total_frames // 2
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
            
            # Read frame
            ret, frame = self.video_cap.read()
            if ret:
                self.current_frame = frame.copy()
                self.display_frame_with_crop()
                
                # Update status
                self.progress_var.set(f"Video loaded: {width}x{height}, {self.total_duration:.1f}s, {self.video_fps:.1f} FPS")
            else:
                messagebox.showerror("Error", "Could not read video frame")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error loading video: {str(e)}")
    
    def display_frame_with_crop(self):
        """Display current frame with crop ratio indicator"""
        if self.current_frame is None:
            return
        
        # Get canvas size
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        if canvas_width <= 1 or canvas_height <= 1:
            # Canvas not ready yet, try again later
            self.root.after(100, self.display_frame_with_crop)
            return
        
        # Resize frame to fit canvas while maintaining aspect ratio
        frame_height, frame_width = self.current_frame.shape[:2]
        scale = min(canvas_width / frame_width, canvas_height / frame_height)
        
        new_width = int(frame_width * scale)
        new_height = int(frame_height * scale)
        
        # Resize frame
        frame_resized = cv2.resize(self.current_frame, (new_width, new_height))
        
        # Convert to PIL and then to PhotoImage
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(frame_rgb)
        self.photo = ImageTk.PhotoImage(image_pil)
        
        # Clear canvas and display image
        self.canvas.delete("all")
        
        # Center image on canvas
        x_offset = (canvas_width - new_width) // 2
        y_offset = (canvas_height - new_height) // 2
        
        self.canvas.create_image(x_offset, y_offset, anchor=tk.NW, image=self.photo)
        
        # Draw crop line
        crop_y = y_offset + int(new_height * self.crop_ratio.get())
        self.crop_line_id = self.canvas.create_line(
            x_offset, crop_y, x_offset + new_width, crop_y,
            fill="red", width=3, tags="crop_line"
        )
        
        # Add crop ratio text
        self.canvas.create_text(
            x_offset + new_width - 10, crop_y - 15,
            text=f"Crop: {self.crop_ratio.get():.2f}",
            fill="red", font=("Arial", 12, "bold"),
            anchor="ne", tags="crop_text"
        )
        
        # Store display info for mouse interactions
        self.display_info = {
            'x_offset': x_offset,
            'y_offset': y_offset,
            'width': new_width,
            'height': new_height
        }
        
        # Bind mouse events for dragging crop line
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
    
    def on_canvas_click(self, event):
        """Handle canvas click for crop line dragging"""
        if hasattr(self, 'display_info'):
            y_in_image = event.y - self.display_info['y_offset']
            if 0 <= y_in_image <= self.display_info['height']:
                new_ratio = y_in_image / self.display_info['height']
                new_ratio = max(0.1, min(0.8, new_ratio))  # Clamp to valid range
                self.crop_ratio.set(new_ratio)
    
    def on_canvas_drag(self, event):
        """Handle canvas drag for crop line dragging"""
        self.on_canvas_click(event)  # Same logic as click
    
    def update_crop_preview(self, value=None):
        """Update crop preview when scale changes"""
        self.display_frame_with_crop()
    
    def on_crop_ratio_change(self, *args):
        """Called when crop ratio changes"""
        self.crop_value_label.config(text=f"{self.crop_ratio.get():.2f}")
        if hasattr(self, 'current_frame') and self.current_frame is not None:
            self.display_frame_with_crop()
    
    def start_extraction(self):
        """Start the extraction process in a separate thread"""
        if self.is_processing:
            return
        
        # Validate inputs
        video_path = self.video_path.get()
        if not video_path or not os.path.exists(video_path):
            messagebox.showerror("Error", "Please select a valid video file")
            return
        
        # Start processing
        self.is_processing = True
        self.process_button.config(state='disabled', text='Processing...')
        self.progress_bar.config(mode='indeterminate')
        self.progress_bar.start()
        self.progress_var.set("Starting extraction process...")
        
        # Start extraction in separate thread
        thread = threading.Thread(target=self.run_extraction, daemon=True)
        thread.start()
    
    def run_extraction(self):
        """Run the extraction process"""
        try:
            video_path = self.video_path.get()
            
            # Call main extraction function with parameters
            self.update_progress("Extracting screenshots...")
            
            custom_folder_name = self.custom_title.get() if self.custom_title.get().strip() else None
            
            result = extract_screenshots(
                video_path=video_path,
                start_time=self.start_time.get(),
                interval=self.interval.get(),
                detection_method=self.method.get(),
                change_threshold=self.change_threshold.get(),
                force_recapture=self.recapture.get(),
                custom_folder_name=custom_folder_name,
                disable_ocr=self.disable_ocr.get(),
                disable_duplicate_detection=self.disable_duplicate_detection.get()
            )
            
            if isinstance(result, tuple) and len(result) == 3:
                success, ascii_folder_name, display_folder_name = result
            else:
                success = result if isinstance(result, bool) else False
                ascii_folder_name = Path(video_path).stem
                display_folder_name = ascii_folder_name
            
            if success and self.create_pdf.get():
                self.update_progress("Creating PDF...")
                pdf_created = self.create_pdf_from_screenshots(ascii_folder_name)
                if pdf_created:
                    self.update_progress("✅ PDF created successfully! Opening...")
                else:
                    self.update_progress("⚠️ PDF creation failed, but screenshots are ready")
            
            if success:
                self.update_progress("✅ Extraction completed successfully!")
                self.root.after(0, self.on_extraction_complete, ascii_folder_name)
            else:
                self.update_progress("❌ Extraction failed!")
                self.root.after(0, self.on_extraction_error, "Extraction process failed")
                
        except Exception as e:
            self.root.after(0, self.on_extraction_error, str(e))
    
    def create_pdf_from_screenshots(self, folder_name):
        """Create PDF from extracted screenshots"""
        try:
            # Determine source directory
            scores_dir = os.path.join(os.getcwd(), "scores")  # Save in project directory
            main_folder = os.path.join(scores_dir, folder_name)
            
            result_dir = os.path.join(main_folder, "screenshots", "result")
            screenshots_dir = os.path.join(main_folder, "screenshots")
            
            if os.path.exists(result_dir) and len(os.listdir(result_dir)) > 0:
                pdf_source_dir = result_dir
            elif os.path.exists(screenshots_dir) and len(os.listdir(screenshots_dir)) > 0:
                pdf_source_dir = screenshots_dir
            else:
                raise Exception(f"No screenshots found in {result_dir} or {screenshots_dir}")
            
            # Import create_pdf module
            code_dir = os.path.dirname(os.path.abspath(__file__))
            sys.path.append(code_dir)
            
            import importlib.util
            create_pdf_path = os.path.join(code_dir, "create_pdf.py")
            spec = importlib.util.spec_from_file_location("create_pdf", create_pdf_path)
            if spec is not None and spec.loader is not None:
                create_pdf_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(create_pdf_module)
                
                # Prepare PDF parameters
                pdf_title = self.custom_title.get() if self.custom_title.get().strip() else folder_name
                pdf_display_title = None if self.skip_pdf_title.get() else pdf_title
                
                # Sanitize filename
                def sanitize_filename(filename):
                    import re
                    return re.sub(r'[<>:"/\\|?*]', '', filename)
                
                pdf_title_safe = sanitize_filename(pdf_title)
                pdf_filename = f"{pdf_title_safe}_score.pdf"
                pdf_path = os.path.join(main_folder, pdf_filename)
                
                # Create PDF
                success = create_pdf_module.create_pdf_from_screenshots(
                    pdf_source_dir,
                    pdf_path,
                    self.crop_ratio.get(),
                    self.strips_per_page.get(),
                    pdf_display_title
                )
                
                if success:
                    self.pdf_path = pdf_path
                    return True
                else:
                    raise Exception("PDF creation failed")
        
        except Exception as e:
            return False
    
    def update_progress(self, message):
        """Update progress message safely from any thread"""
        def update():
            self.progress_var.set(message)
        self.root.after(0, update)
    
    def on_extraction_complete(self, folder_name):
        """Called when extraction completes successfully"""
        self.is_processing = False
        self.process_button.config(state='normal', text='🎬 Start Extraction')
        self.progress_bar.stop()
        self.progress_bar.config(mode='determinate', value=100)
        
        message = "Extraction completed successfully!"
        if self.create_pdf.get() and hasattr(self, 'pdf_path'):
            pdf_folder = os.path.dirname(self.pdf_path)
            pdf_name = os.path.basename(self.pdf_path)
            message += f"\n\n📄 PDF created: {pdf_name}"
            message += f"\n📁 Location: {pdf_folder}"
            
            # Automatically open the PDF
            try:
                # Try different methods based on OS
                if os.name == 'nt':  # Windows
                    os.startfile(self.pdf_path)
                    message += "\n\n✅ PDF opened automatically!"
                elif os.name == 'posix':  # macOS and Linux
                    if sys.platform == 'darwin':  # macOS
                        subprocess.run(['open', self.pdf_path], check=False)
                    else:  # Linux
                        subprocess.run(['xdg-open', self.pdf_path], check=False)
                    message += "\n\n✅ PDF opened automatically!"
                else:
                    # Fallback for other systems
                    import webbrowser
                    webbrowser.open(f'file://{os.path.abspath(self.pdf_path)}')
                    message += "\n\n✅ PDF opened in browser!"
                    
            except Exception as e:
                message += f"\n\n⚠️ Could not open PDF automatically: {str(e)}"
                
            # Add option to open folder
            message += f"\n\n📂 Click OK to also open the folder containing the PDF"
            messagebox.showinfo("Success", message)
            
            # Open folder containing the PDF
            try:
                if os.name == 'nt':  # Windows
                    subprocess.run(['explorer', '/select,', self.pdf_path], check=False)
                elif os.name == 'posix':
                    if sys.platform == 'darwin':  # macOS
                        subprocess.run(['open', '-R', self.pdf_path], check=False)
                    else:  # Linux
                        subprocess.run(['xdg-open', pdf_folder], check=False)
            except Exception as e:
                pass
                
        else:
            messagebox.showinfo("Success", message)
    
    def on_extraction_error(self, error_message):
        """Called when extraction encounters an error"""
        self.is_processing = False
        self.process_button.config(state='normal', text='🎬 Start Extraction')
        self.progress_bar.stop()
        self.progress_bar.config(mode='determinate', value=0)
        self.progress_var.set("Ready to start extraction...")
        
        messagebox.showerror("Error", f"Extraction failed:\n{error_message}")

def main():
    """Main entry point for the GUI application"""
    try:
        # Try to use tkinterdnd2 for better drag-drop support
        try:
            from tkinterdnd2 import TkinterDnD
            root = TkinterDnD.Tk()
        except ImportError:
            # Fallback to regular tkinter
            root = tk.Tk()
        
        # Configure style for better appearance
        style = ttk.Style()
        
        # Try to use native theme for proper checkmarks, fallback to others
        available_themes = style.theme_names()
        if 'vista' in available_themes:  # Windows Vista/7/8/10/11 native theme
            style.theme_use('vista')
        elif 'winnative' in available_themes:  # Windows native theme
            style.theme_use('winnative')
        elif 'aqua' in available_themes:  # macOS native theme
            style.theme_use('aqua')
        else:
            style.theme_use('clam')  # Fallback to clam theme
        
        # Configure checkbox style to show proper checkmarks
        try:
            # Configure checkbox to show checkmarks instead of crosses
            style.configure('TCheckbutton', focuscolor='none')
            # Map the checkbox states to show proper symbols
            style.map('TCheckbutton',
                     indicatorcolor=[('selected', '#0078d4'),
                                   ('!selected', 'white')])
        except Exception:
            # Fallback if style configuration fails
            pass
        
        # Create and run the application
        app = VideoExtractorGUI(root)
        
        # Handle window closing
        def on_closing():
            if app.video_cap:
                app.video_cap.release()
            root.destroy()
        
        root.protocol("WM_DELETE_WINDOW", on_closing)
        
        # Center window on screen
        root.update_idletasks()
        width = root.winfo_width()
        height = root.winfo_height()
        x = (root.winfo_screenwidth() // 2) - (width // 2)
        y = (root.winfo_screenheight() // 2) - (height // 2)
        root.geometry(f"{width}x{height}+{x}+{y}")
        
        root.mainloop()
        
    except Exception as e:
        print(f"Error starting GUI: {e}")
        messagebox.showerror("Error", f"Failed to start application:\n{str(e)}")

if __name__ == "__main__":
    main()