#!/usr/bin/env python3
"""
Wrapper script to run the piano score extraction from the reorganized code folder.
This makes it easier to run the extraction without having to change directories.
"""
import os
import sys
import subprocess

def main():
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to the main.py script in the code folder
    main_script = os.path.join(script_dir, "code", "main.py")
    
    if not os.path.exists(main_script):
        print(f"Error: Could not find main.py at {main_script}")
        sys.exit(1)
    
    # Forward all command line arguments to the main script
    cmd = [sys.executable, main_script] + sys.argv[1:]
    
    try:
        # Change to the code directory to ensure relative paths work correctly
        os.chdir(os.path.join(script_dir, "code"))
        
        # Run the main script
        result = subprocess.run(cmd, cwd=os.path.join(script_dir, "code"))
        
        # Exit with the same code as the main script
        sys.exit(result.returncode)
        
    except FileNotFoundError:
        print(f"Error: Python executable not found: {sys.executable}")
        sys.exit(1)
    except Exception as e:
        print(f"Error running extraction: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()