import { useState, useEffect } from 'react';
import './App.css';

const API_URL = 'http://localhost:8000';

function App() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [stats, setStats] = useState(null);
  const [examples, setExamples] = useState([]);
  const [useAI, setUseAI] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchStats();
    fetchExamples();
  }, []);

  const fetchStats = async () => {
    try {
      const response = await fetch(`${API_URL}/stats`);
      const data = await response.json();
      setStats(data);
    } catch (err) {
      console.error('Error fetching stats:', err);
    }
  };

  const fetchExamples = async () => {
    try {
      const response = await fetch(`${API_URL}/examples`);
      const data = await response.json();
      setExamples(data.examples);
    } catch (err) {
      console.error('Error fetching examples:', err);
    }
  };

  const handleSearch = async (searchQuery = query) => {
    if (!searchQuery.trim()) {
      setError('Please enter a search query');
      return;
    }

    setLoading(true);
    setError(null);
    setResults(null);

    try {
      const response = await fetch(`${API_URL}/search`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: searchQuery,
          k: 5,
          use_ai: useAI
        })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Search failed');
      }

      const data = await response.json();
      setResults(data);
    } catch (err) {
      setError(err.message || 'Search failed. Please try again.');
      console.error('Search error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleExampleClick = (example) => {
    setQuery(example);
    handleSearch(example);
  };

  return (
    <div className="app">
      <div className="container">
        {/* Header */}
        <header className="header">
          <div className="header-content">
            <h1 className="title">🏨 Hotel Finder</h1>
            <p className="subtitle">AI-Powered Hotel Recommendation System</p>
          </div>
          
          {stats && (
            <div className="stats-bar">
              <div className="stat-item">
                <span className="stat-label">Total Hotels</span>
                <span className="stat-value">{stats.total_hotels}</span>
              </div>
              <div className="stat-item">
                <span className="stat-label">Price Range</span>
                <span className="stat-value">₹{stats.price_range.min} - ₹{stats.price_range.max}</span>
              </div>
              <div className="stat-item">
                <span className="stat-label">Avg Price</span>
                <span className="stat-value">₹{Math.round(stats.price_range.avg)}</span>
              </div>
            </div>
          )}
        </header>

        {/* Search Section */}
        <div className="search-section">
          <div className="search-container">
            <input
              type="text"
              className="search-input"
              placeholder="e.g., 'Cheapest hotels in Mumbai' or 'Hotels under 1500 rupees'"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
            />
            <button 
              className="search-button"
              onClick={() => handleSearch()}
              disabled={loading}
            >
              {loading ? '🔄 Searching...' : '🔍 Search'}
            </button>
          </div>

          <div className="ai-toggle">
            <label className="toggle-label">
              <input
                type="checkbox"
                checked={useAI}
                onChange={(e) => setUseAI(e.target.checked)}
              />
              <span className="toggle-text">🤖 Use AI Generation</span>
            </label>
          </div>
        </div>

        {/* Example Queries */}
        <div className="examples-section">
          <h3 className="examples-title">💡 Try these examples:</h3>
          <div className="examples-grid">
            {examples.map((example, idx) => (
              <button
                key={idx}
                className="example-chip"
                onClick={() => handleExampleClick(example)}
              >
                {example}
              </button>
            ))}
          </div>
        </div>

        {/* Error Message */}
        {error && (
          <div className="error-message">
            ❌ {error}
          </div>
        )}

        {/* AI Response */}
        {results?.ai_response && (
          <div className="ai-response">
            <h3 className="ai-response-title">🤖 AI Recommendation</h3>
            <p className="ai-response-text">{results.ai_response}</p>
          </div>
        )}

        {/* Results */}
        {results && results.hotels.length > 0 && (
          <div className="results-section">
            <div className="results-header">
              <h2 className="results-title">
                📊 Found {results.total_found} hotels
                {results.filters_applied && ' (filtered by your criteria)'}
              </h2>
            </div>

            <div className="hotels-grid">
              {results.hotels.map((item, idx) => {
                const hotel = item.hotel;
                return (
                  <div key={idx} className="hotel-card">
                    <div className="hotel-header">
                      <h3 className="hotel-name">{hotel.Hotel_name}</h3>
                      <div className="relevance-badge">
                        ⭐ {Math.round(item.score * 100)}% match
                      </div>
                    </div>

                    <div className="hotel-details">
                      <div className="detail-row">
                        <span className="detail-icon">📍</span>
                        <span className="detail-text">{hotel.Location}</span>
                      </div>

                      <div className="detail-row">
                        <span className="detail-icon">💰</span>
                        <span className="detail-text price">
                          ₹{hotel.Price?.toLocaleString()}/night
                        </span>
                      </div>

                      {hotel.Rating > 0 && (
                        <div className="detail-row">
                          <span className="detail-icon">👥</span>
                          <span className="detail-text">
                            {hotel.Rating} reviews
                            {hotel.Rating >= 500 && ' 🔥'}
                          </span>
                        </div>
                      )}

                      {hotel.Discount && (
                        <div className="detail-row discount">
                          <span className="detail-icon">🎯</span>
                          <span className="detail-text">{hotel.Discount}</span>
                        </div>
                      )}
                    </div>

                    <button className="book-button">
                      View Details
                    </button>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* No Results */}
        {results && results.hotels.length === 0 && (
          <div className="no-results">
            <div className="no-results-icon">🔍</div>
            <h3>No hotels found</h3>
            <p>{results.message || 'Try adjusting your search criteria'}</p>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;