#!/bin/bash
# AI Product Expert Bot - Streamlit Web Interface Launcher
# Usage: ./run_streamlit.sh

echo "🛍️ Starting AI Product Expert Bot Web Interface..."
echo "=================================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run:"
    echo "   python3 -m venv venv"
    echo "   source venv/bin/activate"
    echo "   pip install -r requirements.txt"
    echo "   pip install streamlit"
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  Warning: .env file not found. Please create it with your API keys."
    echo "   Copy .env.example to .env and fill in your credentials."
fi

# Check streamlit installation
if ! command -v streamlit &> /dev/null; then
    echo "❌ Streamlit not installed. Installing now..."
    pip install streamlit
fi

echo "🚀 Launching Streamlit app..."
echo "   The app will open in your browser at http://localhost:8501"
echo "   Press Ctrl+C to stop the server"
echo ""

# Launch Streamlit with custom configuration
streamlit run streamlit_app.py \
    --server.port 8501 \
    --server.address localhost \
    --server.headless false \
    --browser.gatherUsageStats false \
    --theme.base light