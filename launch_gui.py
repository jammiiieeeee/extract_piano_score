#!/usr/bin/env python3
"""
Simple launcher for the Video Extractor GUI
Run this file to start the graphical user interface
"""

import sys
import os

# Add current directory and code directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
code_dir = os.path.join(current_dir, "code")
sys.path.append(current_dir)
sys.path.append(code_dir)

if __name__ == "__main__":
    try:
        from video_extractor_gui import main
        main()
    except Exception as e:
        print(f"Error launching GUI: {e}")
        print("\nTrying to install required packages...")
        try:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pillow"])
            print("✓ Pillow installed successfully")
            print("✓ Now trying to launch GUI again...")
            from video_extractor_gui import main
            main()
        except Exception as e2:
            print(f"Still failed: {e2}")
            print("Please ensure you have Python tkinter and PIL/Pillow installed")
            input("Press Enter to exit...")