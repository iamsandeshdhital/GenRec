# GenRec API Documentation

## Base URL
```
http://localhost:5000
```

## Endpoints

### GET /
Web interface home page.

### GET /demo
Run a demo recommendation and view results.

### POST /recommend
Generate recommendations for a user.

**Request Body:**
```json
{
  "user_data": {
    "user_id": "user_123",
    "age_group": "25-34",
    "gender": "M",
    "language": "English",
    "subscription_tier": "Premium",
    "account_age_days": 730,
    "preferred_genres": ["Sci-Fi", "Drama"]
  },
  "interactions": [
    {
      "title": "Stranger Things",
      "type": "series",
      "genre": "Sci-Fi",
      "user_rating": 5,
      "watch_percentage": 95,
      "timestamp": "2024-11-01"
    }
  ],
  "candidates": [
    {
      "id": "item_1",
      "title": "Inception",
      "type": "movie",
      "genres": ["Sci-Fi", "Action"],
      "year": 2010,
      "rating": 8.8,
      "duration_minutes": 148,
      "description": "A thief who steals secrets through dream-sharing technology."
    }
  ],
  "context": {
    "time_of_day": "evening",
    "day_of_week": "Friday",
    "device": "Smart TV",
    "location_type": "home",
    "session_duration_minutes": 45,
    "items_viewed": 8,
    "items_clicked": 3,
    "recent_searches": ["sci-fi series"]
  }
}
```

**Response:**
```json
{
  "recommendations": ["item_1", "item_9", "item_11"],
  "scores": {
    "item_1": 1.0,
    "item_9": 0.95,
    "item_11": 0.9
  },
  "latency_ms": 2.03,
  "cache_hit": false,
  "metadata": {
    "cached": false,
    "candidates_considered": 15,
    "ranking_latency_ms": 1.5
  }
}
```

### GET /api/health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "cache_size": 5,
  "metrics": {
    "total_requests": 10,
    "cache_hits": 3,
    "cache_hit_rate": 0.3,
    "average_latency_ms": 2.5,
    "cache_size": 5
  }
}
```

### GET /api/metrics
Get serving metrics.

**Response:**
```json
{
  "total_requests": 100,
  "cache_hits": 30,
  "cache_hit_rate": 0.3,
  "average_latency_ms": 2.5,
  "cache_size": 10
}
```

### POST /api/evaluate
Evaluate rankings against relevance grades.

**Request Body:**
```json
{
  "rankings": [
    ["item_1", "item_2", "item_3"],
    ["item_4", "item_5", "item_6"]
  ],
  "relevance_grades": [
    {"item_1": 3, "item_2": 2, "item_3": 1},
    {"item_4": 2, "item_5": 3, "item_6": 1}
  ]
}
```

**Response:**
```json
{
  "ndcg": 0.92,
  "mrr": 0.85,
  "map": 0.78,
  "precision": {"5": 0.6, "10": 0.5},
  "recall": {"5": 0.4, "10": 0.6},
  "coverage": 0.85,
  "diversity": 0.72,
  "novelty": 3.5
}
```

## Error Responses

All errors return JSON with appropriate HTTP status codes:

```json
{
  "error": "Description of the error"
}
```

| Status Code | Description |
|-------------|-------------|
| 400 | Bad Request - Invalid input |
| 404 | Not Found - Resource not found |
| 500 | Internal Server Error |
