# 🛍️ AI Product Expert Bot - Streamlit Web Interface

A beautiful, user-friendly web interface for the AI Product Expert Bot that allows you to search for products using natural language queries in Danish or English.

## ✨ Features

### 🔍 **Smart Search**
- **Natural Language**: Search in Danish or English
- **Price Filters**: "under 500 kroner", "less than $100"
- **Semantic Understanding**: Finds related products even with different terms
- **Image Upload**: Enhanced search with visual similarity (optional)

### 🎯 **Hybrid Search Technology**
- **Dense Vector Search**: Semantic similarity using AI embeddings
- **Sparse Vector Search**: Keyword matching for precise terms
- **Product Type Matching**: Intelligent category recognition
- **Price Range Filtering**: Automatic price constraint detection

### 📊 **Rich Results Display**
- **Product Cards**: Images, prices, availability, brand info
- **Relevance Scores**: See how well each result matches your query
- **Direct Links**: Click through to view products on retailer site
- **Interactive Interface**: Hover, expand, and explore results

### 🛠️ **Developer Features**
- **Debug Mode**: View raw search results and performance metrics
- **Search History**: Quick access to recent queries
- **Database Statistics**: Real-time collection info
- **Error Handling**: Graceful failure with helpful messages

## 🚀 Quick Start

### 1. **Launch the Interface**
```bash
# Option 1: Use the launcher script
./run_streamlit.sh

# Option 2: Manual launch
source venv/bin/activate
streamlit run streamlit_app.py
```

### 2. **Open Your Browser**
The interface will automatically open at: `http://localhost:8501`

### 3. **Start Searching!**
Try these example searches:
- `"stegepande under 500 kroner"` (frying pan under 500 DKK)
- `"Tefal kettle"` (Tefal kettles)
- `"Ordo tandbørste"` (Ordo toothbrush)

## 🎨 Interface Overview

### **Main Search Area**
- **Search Input**: Enter your query in natural language
- **Image Upload**: Optional image for visual similarity search
- **Example Buttons**: Quick access to common searches

### **Sidebar Controls**
- **Search Settings**: Adjust number of results
- **Database Info**: See collection statistics
- **Advanced Options**: Debug mode, raw results view
- **Search History**: Revisit recent queries

### **Results Display**
- **Product Cards**: Rich display with images and details
- **Metrics**: Relevance scores and search performance
- **Product Links**: Direct access to retailer pages

## 🔧 Configuration

### **Required Environment Variables**
Ensure your `.env` file contains:
```env
# Qdrant Vector Database
QDRANT_URL=https://your-cluster.qdrant.io:6333
QDRANT_API_KEY=your_api_key

# Google Cloud Platform
GOOGLE_CLOUD_PROJECT=your_project_id
GOOGLE_CLOUD_REGION=us-central1

# Gemini AI
GEMINI_API_KEY=your_gemini_key
```

### **Optional Settings**
```env
# Logging and Performance
LOG_LEVEL=INFO
MAX_BATCH_SIZE=10
MAX_IMAGE_SIZE_MB=10
```

## 📝 Usage Examples

### **Danish Queries**
- `"kaffemaskine under 400 kroner"` - Coffee machine under 400 DKK
- `"stegepande Tefal sort"` - Black Tefal frying pan
- `"tandbørste til børn"` - Toothbrush for children

### **English Queries**
- `"coffee machine under 400 kr"` - Coffee machine under 400 DKK
- `"Tefal black frying pan"` - Black Tefal frying pan
- `"kids toothbrush"` - Children's toothbrush

### **Advanced Searches**
- `"kitchen scale under 300 DKK in stock"` - With availability filter
- `"OFYR grill accessories"` - Brand-specific search
- Upload an image + "similar products"` - Visual similarity search

## 🎯 Search Tips

### **Get Better Results**
1. **Be Specific**: Include brand, color, size when known
2. **Use Price Ranges**: "under X kroner", "between X and Y DKK"
3. **Try Both Languages**: Danish and English work equally well
4. **Include Images**: Upload product photos for visual similarity

### **Understanding Scores**
- **0.3-1.0**: Excellent match (exact or very similar products)
- **0.1-0.3**: Good match (related products, similar category)
- **0.0-0.1**: Weak match (loosely related or filtered results)

## 🔍 Troubleshooting

### **No Results Found**
- Try broader search terms (e.g., "pan" instead of "stegepande")
- Check if products exist in the database category
- Verify price ranges are reasonable for the product type

### **Slow Performance**
- Reduce number of results in sidebar
- Check your internet connection
- Verify API keys are correctly configured

### **Connection Errors**
- Check `.env` file configuration
- Verify Qdrant cluster is accessible
- Ensure Google Cloud credentials are valid

## 📊 Performance Metrics

The interface displays real-time metrics:
- **Search Time**: How long the query took to process
- **Results Count**: Number of products found
- **Average Relevance**: Quality of matches
- **Database Status**: Collection health and size

## 🎨 Customization

### **Styling**
The interface uses custom CSS for a polished look:
- **Gradient Header**: Eye-catching brand presentation
- **Card Layout**: Clean, organized result display
- **Responsive Design**: Works on desktop and mobile
- **Color Coding**: Visual indicators for availability, relevance

### **Adding Features**
To extend the interface:
1. **New Search Parameters**: Add sliders/inputs to sidebar
2. **Enhanced Filters**: Create dropdown menus for categories
3. **Analytics**: Add charts for search patterns
4. **Export Options**: Save results to CSV/PDF

## 🚀 Deployment Options

### **Local Development**
```bash
streamlit run streamlit_app.py --server.port 8501
```

### **Production Deployment**
```bash
# Docker deployment
streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8080

# Cloud deployment (Streamlit Cloud, Heroku, etc.)
# Use streamlit_app.py as entry point
```

## 📈 Future Enhancements

### **Planned Features**
- [ ] **Product Comparison**: Side-by-side product analysis
- [ ] **Favorites System**: Save interesting products
- [ ] **Search Analytics**: Track popular queries and results
- [ ] **Multi-language UI**: Full interface translation
- [ ] **Voice Search**: Speech-to-text query input
- [ ] **Recommendation Engine**: "Customers also viewed"

### **Technical Improvements**
- [ ] **Caching**: Redis-based result caching
- [ ] **Load Balancing**: Handle multiple concurrent users
- [ ] **A/B Testing**: Compare different search algorithms
- [ ] **Real-time Updates**: Live product availability status

---

## 🎉 Enjoy Searching!

The AI Product Expert Bot Streamlit interface provides a powerful, intuitive way to explore your product database. Whether you're searching in Danish or English, looking for specific brands or browsing categories, the intelligent search system will help you find exactly what you need.

**Happy Shopping!** 🛒✨