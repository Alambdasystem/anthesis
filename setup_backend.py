"""
Setup script for Anthesis Backend
Run this first: python setup_backend.py
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required Python packages"""
    requirements = [
        'flask',
        'flask-cors',
        'sqlite3'  # Built into Python, but listed for completeness
    ]
    
    print("Installing backend dependencies...")
    
    for package in requirements:
        if package == 'sqlite3':
            continue  # Skip sqlite3 as it's built-in
            
        try:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"✅ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")
            return False
    
    return True

def create_startup_script():
    """Create startup scripts for different platforms"""
    
    # Windows batch file
    with open('start_backend.bat', 'w') as f:
        f.write("""@echo off
echo Starting Anthesis Backend...
python anthesis_backend.py
pause
""")
    
    # Unix/Linux shell script
    with open('start_backend.sh', 'w') as f:
        f.write("""#!/bin/bash
echo "Starting Anthesis Backend..."
python3 anthesis_backend.py
""")
    
    # Make shell script executable on Unix systems
    try:
        os.chmod('start_backend.sh', 0o755)
    except:
        pass  # Ignore on Windows
    
    print("✅ Startup scripts created:")
    print("   - start_backend.bat (Windows)")
    print("   - start_backend.sh (Unix/Linux)")

def main():
    print("=== Anthesis Backend Setup ===")
    print()
    
    # Install requirements
    if install_requirements():
        print("\n✅ All dependencies installed successfully!")
    else:
        print("\n❌ Some dependencies failed to install. Please install manually:")
        print("   pip install flask flask-cors")
        return
    
    # Create startup scripts
    create_startup_script()
    
    print("\n=== Setup Complete! ===")
    print("\nTo start the backend:")
    print("1. Run: python anthesis_backend.py")
    print("2. Or use: start_backend.bat (Windows) / ./start_backend.sh (Unix)")
    print("\nThe backend will be available at: http://localhost:5000")
    print("Health check: http://localhost:5000/api/health")
    print("\nUpdate your frontend to use: http://localhost:5000 as the API base URL")

if __name__ == '__main__':
    main()
