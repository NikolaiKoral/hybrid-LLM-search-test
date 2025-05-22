# 🚀 How to Start the AI Product Expert Bot Web Interface

## Quick Start (Manual)

### 1. Open Terminal
Navigate to your project directory:
```bash
cd "/Users/nikolailind/Documents/GitHub/Test hybrid"
```

### 2. Activate Virtual Environment
```bash
source venv/bin/activate
```

### 3. Start the App
**Option A - Simple Test Version:**
```bash
streamlit run simple_app.py --server.port 8502
```

**Option B - Full Interface:**
```bash
streamlit run streamlit_app.py --server.port 8503
```

### 4. Open Browser
The terminal will show a message like:
```
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8502
```

**Manually open:** http://localhost:8502 (or 8503 for full app)

## 🔧 Troubleshooting

### Problem: "Connection Refused"
**Solution:**
1. Make sure the terminal shows "You can now view your Streamlit app"
2. Wait 10-15 seconds for full startup
3. Try refreshing the browser page
4. Check the correct port number (8502 or 8503)

### Problem: "Module Not Found"
**Solution:**
```bash
# Reinstall dependencies
source venv/bin/activate
pip install -r requirements.txt
pip install streamlit
```

### Problem: "Configuration Error"
**Solution:**
1. Check your `.env` file exists
2. Verify all API keys are set correctly
3. Test with: `python -c "import config; config.validate_config()"`

## 🎯 What to Expect

### Simple Test App (port 8502)
- Basic interface to test functionality
- Verifies all modules load correctly
- Simple search input for testing

### Full Interface (port 8503)
- Complete product search interface
- Image upload capability
- Advanced filters and settings
- Rich product result cards

## 📱 Using the Interface

### Search Examples
Try these queries in the search box:
- `"stegepande under 500 kroner"` (Danish)
- `"Tefal kettle"` (English)
- `"Ordo tandbørste"` (Danish toothbrush)

### Features to Test
1. **Natural Language Search** - Type queries in Danish or English
2. **Price Filtering** - Include price ranges in your search
3. **Image Upload** - Upload product images for visual search
4. **Result Cards** - Click on products to view details

## 🛑 Stopping the App

Press `Ctrl+C` in the terminal where Streamlit is running.

## 🔄 Restarting

If something goes wrong:
1. Press `Ctrl+C` to stop
2. Wait a few seconds
3. Run the streamlit command again

## 📞 Getting Help

If you're still having issues:

1. **Check Terminal Output** - Look for error messages
2. **Test Imports** - Run: `python -c "import streamlit; print('OK')"`
3. **Try Different Port** - Use `--server.port 8504` if others are busy
4. **Clear Browser Cache** - Hard refresh with Cmd+Shift+R (Mac)

## 🎉 Success!

When working correctly, you should see:
- Beautiful web interface in your browser
- Search box that accepts Danish/English queries
- Product results with images, prices, and details
- Smooth, responsive interface

**Happy searching!** 🛍️