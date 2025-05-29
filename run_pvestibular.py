#!/usr/bin/env python3
"""
pVestibular Analysis Platform Launcher
=====================================

This script provides a convenient way to launch the pVestibular analysis platform
with automatic dependency checking and installation assistance.
"""

import sys
import subprocess
import os
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required!")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        'streamlit',
        'numpy', 
        'scipy',
        'matplotlib',
        'pandas'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} is missing")
    
    return missing_packages

def install_dependencies():
    """Install missing dependencies."""
    try:
        print("📦 Installing dependencies from requirements.txt...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False
    except FileNotFoundError:
        print("❌ requirements.txt not found!")
        return False

def check_files():
    """Check if required files exist."""
    required_files = [
        'pvestibular.py',
        'utils.py', 
        'hit_utils.py',
        'new_plist_parser.py',
        'requirements.txt'
    ]
    
    missing_files = []
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file} found")
        else:
            missing_files.append(file)
            print(f"❌ {file} missing")
    
    return missing_files

def launch_streamlit():
    """Launch the Streamlit application."""
    try:
        print("\n🚀 Launching pVestibular Analysis Platform...")
        print("📝 The application will open in your default web browser")
        print("🔗 URL: http://localhost:8501")
        print("⏹️  Press Ctrl+C to stop the application")
        print("-" * 50)
        
        # Launch streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "pvestibular.py",
            "--server.headless", "false",
            "--server.runOnSave", "true",
            "--theme.base", "light"
        ])
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to launch Streamlit: {e}")
        return False
    except KeyboardInterrupt:
        print("\n\n👋 pVestibular Analysis Platform stopped.")
        return True

def main():
    """Main launcher function."""
    print("🧠 pVestibular Analysis Platform Launcher")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check required files
    missing_files = check_files()
    if missing_files:
        print(f"\n❌ Missing required files: {', '.join(missing_files)}")
        print("Please ensure you have all necessary files in the current directory.")
        sys.exit(1)
    
    # Check dependencies
    missing_packages = check_dependencies()
    
    if missing_packages:
        print(f"\n📦 Missing packages: {', '.join(missing_packages)}")
        response = input("Would you like to install them automatically? (y/n): ")
        
        if response.lower() in ['y', 'yes']:
            if not install_dependencies():
                print("❌ Failed to install dependencies. Please install manually:")
                print("pip install -r requirements.txt")
                sys.exit(1)
        else:
            print("Please install missing dependencies manually:")
            print("pip install -r requirements.txt")
            sys.exit(1)
    
    print("\n✅ All checks passed!")
    
    # Launch the application
    launch_streamlit()

if __name__ == "__main__":
    main() 