#!/usr/bin/env python3
"""
Simple launcher for the Streamlit app
"""

import subprocess
import sys
import os
import time
import webbrowser

def main():
    print("🛍️ AI Product Expert Bot - Streamlit Launcher")
    print("=" * 50)
    
    # Check if we're in a virtual environment
    if not os.path.exists("venv/bin/activate"):
        print("❌ Virtual environment not found!")
        print("Please run: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt")
        sys.exit(1)
    
    # Check if .env exists
    if not os.path.exists(".env"):
        print("⚠️  Warning: .env file not found!")
        print("Please copy .env.example to .env and configure your API keys")
    
    print("🚀 Starting Streamlit server...")
    print("📱 The app will be available at: http://localhost:8501")
    print("🛑 Press Ctrl+C to stop the server")
    print("-" * 50)
    
    # Launch streamlit
    try:
        # Use subprocess to run streamlit
        cmd = [
            "venv/bin/python", "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ]
        
        # Start the process
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        # Wait a moment for server to start
        time.sleep(3)
        
        # Try to open browser
        try:
            webbrowser.open("http://localhost:8501")
            print("🌐 Opened browser automatically")
        except:
            print("🌐 Please open http://localhost:8501 in your browser")
        
        # Wait for the process
        process.wait()
        
    except KeyboardInterrupt:
        print("\n🛑 Stopping server...")
        if 'process' in locals():
            process.terminate()
        print("✅ Server stopped")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()