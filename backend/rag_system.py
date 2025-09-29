import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict, Any
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

warnings.filterwarnings('ignore')

class HotelRAGSystem:
    def __init__(self, csv_path: str, hf_token: str = None):
        self.csv_path = csv_path
        self.hf_token = hf_token
        self.df = None
        self.documents = []
        self.embed_model = None
        self.doc_embeddings = None
        self.faiss_index = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.generator = None
        self.tokenizer = None
        self.model = None
        
    def load_data(self):
        """Load the OYO dataset"""
        print(f"📊 Loading dataset from {self.csv_path}...")
        self.df = pd.read_csv(self.csv_path)
        self.df = self.df.reset_index(drop=True)
        print(f"Loaded {len(self.df)} hotels")
        return {
            "total_hotels": len(self.df),
            "columns": list(self.df.columns),
            "price_range": {
                "min": float(self.df['Price'].min()),
                "max": float(self.df['Price'].max()),
                "avg": float(self.df['Price'].mean())
            }
        }
    
    def create_documents(self):
        """Create smart documents for RAG"""
        print("📝 Creating smart documents...")
        self.documents = []
        
        for idx, row in self.df.iterrows():
            parts = []
            
            if pd.notna(row['Hotel_name']):
                parts.append(f"Hotel: {row['Hotel_name']}")
            
            if pd.notna(row['Location']):
                location = row['Location']
                parts.append(f"Location: {location}")
                location_lower = location.lower()
                
                if 'mumbai' in location_lower:
                    parts.append("City: Mumbai")
                elif 'delhi' in location_lower:
                    parts.append("City: Delhi")
                elif 'bangalore' in location_lower or 'bengaluru' in location_lower:
                    parts.append("City: Bangalore")
                
                if 'airport' in location_lower:
                    parts.append("Near airport, convenient for travelers")
                if 'metro' in location_lower or 'station' in location_lower:
                    parts.append("Good public transport connectivity")
                if 'andheri' in location_lower:
                    parts.append("Andheri area, business district")
            
            if pd.notna(row['Price']):
                price = row['Price']
                parts.append(f"Price: ₹{price:,.0f}")
                
                if price <= 800:
                    parts.append("Ultra budget accommodation, very cheap")
                elif price <= 1500:
                    parts.append("Budget-friendly option, affordable")
                elif price <= 2500:
                    parts.append("Mid-range hotel, good value")
                elif price <= 3500:
                    parts.append("Premium accommodation, higher quality")
                else:
                    parts.append("Luxury hotel, expensive, high-end")
            
            if pd.notna(row['Rating']) and row['Rating'] > 0:
                rating = int(row['Rating'])
                parts.append(f"Customer reviews: {rating}")
                if rating >= 1000:
                    parts.append("Extremely popular, highly rated")
                elif rating >= 500:
                    parts.append("Very popular among guests")
                elif rating >= 200:
                    parts.append("Well-reviewed, popular choice")
                elif rating >= 50:
                    parts.append("Good customer feedback")
                else:
                    parts.append("Limited reviews, newer property")
            
            if pd.notna(row['Discount']):
                parts.append(f"Discount available: {row['Discount']}")
            
            document = ". ".join(parts)
            self.documents.append(document)
        
        print(f"✅ Created {len(self.documents)} documents")
    
    def setup_retrieval(self):
        """Setup embedding and keyword retrieval"""
        print("🔤 Setting up retrieval systems...")
        
        self.embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.doc_embeddings = self.embed_model.encode(self.documents, show_progress_bar=True)
        
        d = self.doc_embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatL2(d)
        self.faiss_index.add(np.array(self.doc_embeddings).astype('float32'))
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.documents)
        
        print("✅ Retrieval systems ready!")
    
    def setup_generation(self):
        """Setup Hugging Face text generation"""
        print("🤖 Setting up text generation...")
        try:
            model_name = "gpt2"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_length=256,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id,
                device=0 if torch.cuda.is_available() else -1
            )
            print("✅ Generation model ready!")
        except Exception as e:
            print(f"⚠️ Generation setup failed: {e}")
            self.generator = None
    
    def apply_strict_filters(self, query: str) -> pd.DataFrame:
        """Apply strict filters based on query"""
        filtered_df = self.df.copy()
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['cheapest', 'cheap', 'budget', 'affordable']):
            price_threshold = filtered_df['Price'].quantile(0.5)
            filtered_df = filtered_df[filtered_df['Price'] <= price_threshold]
        elif any(word in query_lower for word in ['luxury', 'premium', 'expensive', 'high-end']):
            price_threshold = filtered_df['Price'].quantile(0.7)
            filtered_df = filtered_df[filtered_df['Price'] >= price_threshold]
        
        price_patterns = [
            (r'under\s+(\d+)', lambda x: filtered_df[filtered_df['Price'] <= int(x)]),
            (r'below\s+(\d+)', lambda x: filtered_df[filtered_df['Price'] <= int(x)]),
            (r'less than\s+(\d+)', lambda x: filtered_df[filtered_df['Price'] <= int(x)]),
            (r'maximum\s+(\d+)', lambda x: filtered_df[filtered_df['Price'] <= int(x)])
        ]
        
        for pattern, filter_func in price_patterns:
            match = re.search(pattern, query_lower)
            if match:
                max_price = match.group(1)
                filtered_df = filter_func(max_price)
                break
        
        location_filters = {
            'mumbai': ['mumbai', 'bombay'],
            'delhi': ['delhi', 'new delhi'],
            'bangalore': ['bangalore', 'bengaluru'],
            'andheri': ['andheri'],
            'airport': ['airport', 'international airport'],
            'station': ['station', 'metro station', 'railway'],
        }
        
        for location_key, variants in location_filters.items():
            if any(variant in query_lower for variant in variants):
                location_mask = filtered_df['Location'].str.contains(
                    '|'.join(variants),
                    case=False,
                    na=False
                )
                if location_mask.any():
                    filtered_df = filtered_df[location_mask]
                    break
        
        return filtered_df
    
    def smart_sort(self, query: str, results: List[Dict]) -> List[Dict]:
        """Smart sorting based on query intent"""
        query_lower = query.lower()
        
        if 'cheapest' in query_lower or 'budget' in query_lower:
            results.sort(key=lambda x: (x['hotel']['Price'], -x['score']))
        elif any(word in query_lower for word in ['popular', 'best', 'top rated']):
            results.sort(key=lambda x: (-x['hotel'].get('Rating', 0), -x['score']))
        elif 'luxury' in query_lower or 'premium' in query_lower:
            results.sort(key=lambda x: (-x['hotel']['Price'] * 0.6 - x['score'] * 0.4,))
        else:
            results.sort(key=lambda x: -x['score'])
        
        return results
    
    def search_hotels(self, query: str, k: int = 10) -> Dict[str, Any]:
        """Main search function"""
        filtered_df = self.apply_strict_filters(query)
        
        if len(filtered_df) == 0:
            return {
                'query': query,
                'hotels': [],
                'total_found': 0,
                'message': "No hotels found matching your criteria. Try broader terms."
            }
        
        filtered_indices = filtered_df.index.tolist()
        query_embedding = self.embed_model.encode([query])
        filtered_embeddings = self.doc_embeddings[filtered_indices]
        
        temp_index = faiss.IndexFlatL2(filtered_embeddings.shape[1])
        temp_index.add(filtered_embeddings.astype('float32'))
        
        distances, indices = temp_index.search(
            query_embedding.astype('float32'),
            min(k, len(filtered_indices))
        )
        
        final_indices = [filtered_indices[i] for i in indices[0]]
        scores = 1 / (1 + distances[0])
        
        results = []
        for idx, score in zip(final_indices, scores):
            hotel_data = self.df.iloc[idx].to_dict()
            # Convert numpy types to Python types
            for key, value in hotel_data.items():
                if isinstance(value, (np.integer, np.floating)):
                    hotel_data[key] = float(value)
                elif pd.isna(value):
                    hotel_data[key] = None
            
            results.append({
                'index': int(idx),
                'score': float(score),
                'hotel': hotel_data,
                'document': self.documents[idx]
            })
        
        results = self.smart_sort(query, results)
        
        return {
            'query': query,
            'hotels': results[:k],
            'total_found': len(filtered_df),
            'filters_applied': len(filtered_df) < len(self.df)
        }
    
    def generate_response(self, query: str, search_results: Dict) -> str:
        """Generate AI response"""
        if not self.generator:
            return self.format_results_template(search_results)
        
        try:
            context_parts = []
            hotels = search_results.get('hotels', [])[:3]
            
            for hotel_data in hotels:
                hotel = hotel_data['hotel']
                context_parts.append(
                    f"{hotel['Hotel_name']} in {hotel['Location']} costs ₹{hotel['Price']} per night"
                )
            
            context = ". ".join(context_parts)
            prompt = f"User Query: {query}\nAvailable Hotels: {context}\nRecommendation: Based on your request, I suggest"
            
            generated = self.generator(
                prompt,
                max_new_tokens=100,
                num_return_sequences=1,
                temperature=0.6,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.2
            )
            
            generated_text = generated[0]['generated_text']
            response = generated_text[len(prompt):].strip()
            
            if not response or len(response) < 20:
                return self.format_results_template(search_results)
            
            return response
        except Exception as e:
            print(f"Generation error: {e}")
            return self.format_results_template(search_results)
    
    def format_results_template(self, search_results: Dict) -> str:
        """Template-based formatting"""
        if not search_results['hotels']:
            return search_results.get('message', 'No results found')
        
        hotels = search_results['hotels']
        summary = f"Found {len(hotels)} hotels matching your criteria. "
        
        top_hotel = hotels[0]['hotel']
        summary += f"Top recommendation: {top_hotel['Hotel_name']} at ₹{top_hotel['Price']}/night."
        
        return summary
    
    def initialize(self):
        """Initialize the complete system"""
        self.load_data()
        self.create_documents()
        self.setup_retrieval()
        self.setup_generation()
        print("✅ RAG System fully initialized!")