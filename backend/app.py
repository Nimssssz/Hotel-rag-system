from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import os
from rag_system import HotelRAGSystem

app = FastAPI(title="OYO Hotel RAG API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG system instance
rag_system = None

# Request/Response models
class SearchRequest(BaseModel):
    query: str
    k: Optional[int] = 5
    use_ai: Optional[bool] = True

class SearchResponse(BaseModel):
    query: str
    hotels: List[dict]
    total_found: int
    filters_applied: bool
    ai_response: Optional[str] = None
    message: Optional[str] = None

@app.on_event("startup")
async def startup_event():
    """Initialize RAG system on startup"""
    global rag_system
    csv_path = os.getenv("CSV_PATH", "oyo_rooms.csv")
    hf_token = os.getenv("HF_TOKEN", "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")
    
    print("🚀 Initializing OYO Hotel RAG System...")
    rag_system = HotelRAGSystem(csv_path, hf_token)
    rag_system.initialize()
    print("✅ System ready!")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "OYO Hotel RAG API is running!",
        "version": "1.0.0",
        "status": "healthy"
    }

@app.get("/stats")
async def get_stats():
    """Get dataset statistics"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    return {
        "total_hotels": len(rag_system.df),
        "price_range": {
            "min": float(rag_system.df['Price'].min()),
            "max": float(rag_system.df['Price'].max()),
            "avg": float(rag_system.df['Price'].mean())
        },
        "cities": rag_system.df['Location'].str.extract(r'(Mumbai|Delhi|Bangalore|Bengaluru)', expand=False).value_counts().to_dict()
    }

@app.post("/search", response_model=SearchResponse)
async def search_hotels(request: SearchRequest):
    """Search for hotels based on query"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        # Perform search
        search_results = rag_system.search_hotels(request.query, k=request.k)
        
        # Generate AI response if requested
        ai_response = None
        if request.use_ai and rag_system.generator:
            ai_response = rag_system.generate_response(request.query, search_results)
        
        return SearchResponse(
            query=search_results['query'],
            hotels=search_results['hotels'],
            total_found=search_results['total_found'],
            filters_applied=search_results.get('filters_applied', False),
            ai_response=ai_response,
            message=search_results.get('message')
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/examples")
async def get_example_queries():
    """Get example queries"""
    return {
        "examples": [
            "Cheapest hotels in Mumbai",
            "Hotels under 1500 rupees",
            "Luxury hotels in Delhi",
            "Budget hotels near airport",
            "Most popular hotels in Andheri",
            "Hotels under 1000 in Mumbai",
            "Premium hotels in Bangalore"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)