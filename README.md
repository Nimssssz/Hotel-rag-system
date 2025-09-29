# 🏨 Hotel RAG System

An AI-powered hotel recommendation system built with **FastAPI**, **React**, and **Retrieval-Augmented Generation (RAG)** using Sentence Transformers, FAISS, and Hugging Face models.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![React](https://img.shields.io/badge/React-18.2.0-61dafb.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-009688.svg)

## 🌟 Features

- 🔍 **Intelligent Search**: Natural language queries like "cheapest hotels in Mumbai" or "hotels under 1500 rupees"
- 🤖 **AI-Powered Recommendations**: Uses Hugging Face transformers for generating human-like responses
- ⚡ **Fast Vector Search**: FAISS-powered semantic search with sentence embeddings
- 🎯 **Smart Filtering**: Automatic price, location, and popularity-based filtering
- 📊 **Real-time Results**: Interactive web interface with instant search results
- 🎨 **Modern UI**: Beautiful, responsive React frontend with gradient designs
- 🔄 **Relevance Scoring**: Shows match percentage for each hotel result

## 📸 Screenshots
<img width="1437" height="707" alt="Screenshot 2025-09-29 at 7 27 53 PM" src="https://github.com/user-attachments/assets/cf1855b0-3a0d-4454-acc2-c2b155511b24" />
<img width="1215" height="227" alt="image" src="https://github.com/user-attachments/assets/4a87b20d-6772-4b1e-a452-7d5e986cd6ef" />
<img width="1437" height="710" alt="Screenshot 2025-09-29 at 7 27 08 PM" src="https://github.com/user-attachments/assets/56c09556-de6c-4b27-9ab8-1e2d32f938b0" />



## 🛠️ Tech Stack

### Backend
- **FastAPI** - Modern Python web framework
- **Sentence Transformers** - Semantic embeddings
- **FAISS** - Vector similarity search
- **Hugging Face Transformers** - AI text generation
- **Pandas** - Data processing
- **scikit-learn** - TF-IDF and ML utilities

### Frontend
- **React** - UI framework
- **CSS3** - Modern styling with gradients
- **Fetch API** - HTTP requests

## 📁 Project Structure

```
hotel-rag-system/
├── backend/
│   ├── app.py              # FastAPI application
│   ├── rag_system.py       # RAG implementation
│   ├── requirements.txt    # Python dependencies
│   └── oyo_rooms.csv       # Dataset (not included)
└── frontend/
    ├── package.json        # Node dependencies
    ├── public/
    │   └── index.html
    └── src/
        ├── App.js          # Main React component
        ├── App.css         # Styles
        ├── index.js        # Entry point
        └── index.css       # Global styles
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Node.js 14+
- npm or yarn
- OYO Rooms dataset (CSV file)

### Backend Setup

1. **Clone the repository**
```bash
https://github.com/Nimssssz/Hotel-rag-system.git
cd hotel-rag-system/backend
```

2. **Create virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Add your dataset**
- Place `oyo_rooms.csv` in the `backend/` folder
- Download from: [Kaggle OYO Dataset](https://www.kaggle.com/datasets/sonu1maheshwari/oyo-hotel-rooms)

5. **Run the backend**
```bash
python app.py
```

Backend will run on: `http://localhost:8000`

### Frontend Setup

1. **Navigate to frontend**
```bash
cd frontend
```

2. **Install dependencies**
```bash
npm install
```

3. **Start development server**
```bash
npm start
```

Frontend will open on: `http://localhost:3000`

## 📊 Dataset

This project uses the OYO Rooms dataset containing hotel information including:
- Hotel names
- Locations
- Prices
- Ratings
- Discounts

**Required columns:**
- `Hotel_name`
- `Location`
- `Price`
- `Rating`
- `Discount`

## 🔧 Configuration

### Backend Configuration

Edit `backend/app.py`:
```python
# Change CSV path
csv_path = os.getenv("CSV_PATH", "oyo_rooms.csv")

# Update Hugging Face token
hf_token = os.getenv("HF_TOKEN", "your_token_here")
```

### Frontend Configuration

Edit `frontend/src/App.js`:
```javascript
// Update API URL
const API_URL = 'http://localhost:8000';
```

## 💡 Usage Examples

### Search Queries

```
"Cheapest hotels in Mumbai"
"Hotels under 1500 rupees"
"Luxury hotels in Delhi"
"Budget hotels near airport"
"Most popular hotels in Andheri"
"Hotels under 1000 in Mumbai"
"Premium hotels in Bangalore"
```

### API Usage

**Search Endpoint:**
```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "cheapest hotels in Mumbai",
    "k": 5,
    "use_ai": true
  }'
```

**Get Statistics:**
```bash
curl http://localhost:8000/stats
```

**Example Queries:**
```bash
curl http://localhost:8000/examples
```

## 🎯 API Documentation

### Endpoints

#### `GET /`
Health check endpoint

**Response:**
```json
{
  "message": "OYO Hotel RAG API is running!",
  "version": "1.0.0",
  "status": "healthy"
}
```

#### `GET /stats`
Get dataset statistics

**Response:**
```json
{
  "total_hotels": 1234,
  "price_range": {
    "min": 500,
    "max": 5000,
    "avg": 1800
  },
  "cities": {...}
}
```

#### `POST /search`
Search for hotels

**Request:**
```json
{
  "query": "cheapest hotels in Mumbai",
  "k": 5,
  "use_ai": true
}
```

**Response:**
```json
{
  "query": "cheapest hotels in Mumbai",
  "hotels": [
    {
      "index": 42,
      "score": 0.89,
      "hotel": {
        "Hotel_name": "OYO 123 Budget Stay",
        "Location": "Andheri, Mumbai",
        "Price": 899,
        "Rating": 450,
        "Discount": "20% off"
      },
      "document": "Hotel: OYO 123..."
    }
  ],
  "total_found": 25,
  "filters_applied": true,
  "ai_response": "Based on your request, I suggest..."
}
```

#### `GET /examples`
Get example queries

**Response:**
```json
{
  "examples": [
    "Cheapest hotels in Mumbai",
    "Hotels under 1500 rupees",
    ...
  ]
}
```

## 🧠 How It Works

### RAG Pipeline

1. **Document Creation**: Hotel data is converted into rich text documents with metadata
2. **Embedding**: Documents are embedded using Sentence Transformers (all-MiniLM-L6-v2)
3. **Indexing**: FAISS creates a vector index for fast similarity search
4. **Query Processing**: User queries are filtered and embedded
5. **Retrieval**: Top-k most relevant hotels are retrieved using vector similarity
6. **Generation**: Hugging Face models generate natural language responses
7. **Ranking**: Results are ranked by price, popularity, or relevance

### Smart Filtering

The system applies intelligent filters based on keywords:
- **Price**: "cheap", "budget", "luxury", "under X"
- **Location**: City names, areas, "near airport"
- **Popularity**: "popular", "best", "top rated"

## 🐛 Troubleshooting

### Common Issues

**Backend won't start:**
```bash
# Check Python version
python --version  # Should be 3.8+

# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

**Frontend connection error:**
```bash
# Check if backend is running
curl http://localhost:8000

# Check CORS settings in app.py
```

**Model download issues:**
```bash
# Set HF_HOME environment variable
export HF_HOME=/path/to/cache

# Or use smaller model in rag_system.py
model_name = "distilgpt2"
```

**Port conflicts:**
```bash
# Backend: Change port in app.py
uvicorn.run(app, host="0.0.0.0", port=8001)

# Frontend: Use different port
PORT=3001 npm start
```

## 🚀 Deployment

### Deploy Backend

**Option 1: Railway**
```bash
railway login
railway init
railway up
```

**Option 2: Render**
- Connect GitHub repository
- Set build command: `pip install -r requirements.txt`
- Set start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`

**Option 3: Docker**
```dockerfile
FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Deploy Frontend

**Build for production:**
```bash
cd frontend
npm run build
```

**Option 1: Vercel**
```bash
npm install -g vercel
vercel --prod
```

**Option 2: Netlify**
```bash
npm install -g netlify-cli
netlify deploy --prod --dir=build
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## 🙏 Acknowledgments

- [Sentence Transformers](https://www.sbert.net/) for semantic embeddings
- [FAISS](https://github.com/facebookresearch/faiss) for vector search
- [Hugging Face](https://huggingface.co/) for transformers
- [FastAPI](https://fastapi.tiangolo.com/) for the amazing web framework
- OYO Rooms dataset from [Kaggle](https://www.kaggle.com/)


---

⭐ Star this repo if you find it helpful!

Built with ❤️ using FastAPI and React
