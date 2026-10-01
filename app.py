"""
GenRec Web Interface
Flask application for the LLM-Backed Recommendation Ranker

Based on: "GenRec: An LLM-Backed Recommendation Ranker at Netflix"
Built by: Sandesh Dhital
"""

from flask import Flask, render_template, request, jsonify, session
import json
import os
from datetime import datetime

from genrec.verbalizer import Verbalizer
from genrec.context_engine import ContextEngine
from genrec.ranker import GenRecRanker
from genrec.reward import RewardModel
from genrec.serving import GenRecServer, ServingConfig
from genrec.evaluation import Evaluator

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'genrec-secret-key')

# Initialize components
verbalizer = Verbalizer(max_history=10)
context_engine = ContextEngine()
reward_model = RewardModel()
evaluator = Evaluator()

# Initialize ranker (use OpenAI if API key available, otherwise mock)
use_openai = bool(os.environ.get('OPENAI_API_KEY'))
ranker = GenRecRanker(use_openai=use_openai)

# Initialize server
server = GenRecServer(ranker, ServingConfig(
    max_candidates=50,
    top_k=10,
    cache_ttl_seconds=300,
))


def load_sample_data():
    """Load sample data for demo"""
    catalog = [
        {"id": "item_1", "title": "Stranger Things", "type": "series", "genres": ["Sci-Fi", "Horror", "Drama"], "year": 2016, "rating": 8.7, "duration_minutes": 50, "description": "A group of kids uncover supernatural mysteries in their small town."},
        {"id": "item_2", "title": "The Crown", "type": "series", "genres": ["Drama", "History", "Biography"], "year": 2016, "rating": 8.6, "duration_minutes": 60, "description": "The reign of Queen Elizabeth II from the 1940s to modern times."},
        {"id": "item_3", "title": "Extraction", "type": "movie", "genres": ["Action", "Thriller"], "year": 2020, "rating": 6.7, "duration_minutes": 117, "description": "A black-market mercenary who has nothing to lose is hired to rescue the kidnapped son of an imprisoned international crime lord."},
        {"id": "item_4", "title": "The Witcher", "type": "series", "genres": ["Fantasy", "Action", "Adventure"], "year": 2019, "rating": 8.2, "duration_minutes": 60, "description": "Geralt of Rivia, a solitary monster hunter, struggles to find his place in a world where people often prove more wicked than beasts."},
        {"id": "item_5", "title": "Money Heist", "type": "series", "genres": ["Crime", "Drama", "Thriller"], "year": 2017, "rating": 8.2, "duration_minutes": 70, "description": "An unusual group of robbers attempt to carry out the most perfect robbery in Spanish history."},
        {"id": "item_6", "title": "Bird Box", "type": "movie", "genres": ["Horror", "Thriller", "Drama"], "year": 2018, "rating": 6.6, "duration_minutes": 124, "description": "Five years after an ominous unseen presence drives most of society to suicide, a mother and her two children make a desperate bid to reach safety."},
        {"id": "item_7", "title": "Dark", "type": "series", "genres": ["Sci-Fi", "Drama", "Crime"], "year": 2017, "rating": 8.8, "duration_minutes": 53, "description": "A family saga with a supernatural twist, set in a German town where the disappearance of children exposes the secrets of its residents."},
        {"id": "item_8", "title": "The Queen's Gambit", "type": "series", "genres": ["Drama"], "year": 2020, "rating": 8.5, "duration_minutes": 375, "description": "The story of an orphaned chess prodigy's rise to the top of the chess world while struggling with addiction."},
        {"id": "item_9", "title": "Inception", "type": "movie", "genres": ["Sci-Fi", "Action", "Thriller"], "year": 2010, "rating": 8.8, "duration_minutes": 148, "description": "A thief who steals corporate secrets through dream-sharing technology is given the inverse task of planting an idea into the mind of a CEO."},
        {"id": "item_10", "title": "Breaking Bad", "type": "series", "genres": ["Crime", "Drama", "Thriller"], "year": 2008, "rating": 9.5, "duration_minutes": 49, "description": "A high school chemistry teacher turned methamphetamine producer partners with a former student."},
        {"id": "item_11", "title": "Interstellar", "type": "movie", "genres": ["Sci-Fi", "Adventure", "Drama"], "year": 2014, "rating": 8.7, "duration_minutes": 169, "description": "A team of explorers travel through a wormhole in space in an attempt to ensure humanity's survival."},
        {"id": "item_12", "title": "The Office", "type": "series", "genres": ["Comedy"], "year": 2005, "rating": 9.0, "duration_minutes": 22, "description": "A mockumentary on a group of typical office workers, where the workday consists of ego clashes, inappropriate behavior, and tedium."},
        {"id": "item_13", "title": "Squid Game", "type": "series", "genres": ["Drama", "Thriller", "Action"], "year": 2021, "rating": 8.0, "duration_minutes": 55, "description": "Hundreds of cash-strapped players accept a strange invitation to compete in children's games for a tempting prize."},
        {"id": "item_14", "title": "Dune", "type": "movie", "genres": ["Sci-Fi", "Adventure"], "year": 2021, "rating": 8.0, "duration_minutes": 155, "description": "Feature adaptation of Frank Herbert's science fiction novel about the son of a noble family entrusted with the protection of the most valuable asset in the galaxy."},
        {"id": "item_15", "title": "Wednesday", "type": "series", "genres": ["Comedy", "Fantasy", "Mystery"], "year": 2022, "rating": 8.5, "duration_minutes": 45, "description": "Follows Wednesday Addams' years as a student at Nevermore Academy, where she attempts to master her emerging psychic ability."},
    ]
    
    user_data = {
        "user_id": "user_123",
        "age_group": "25-34",
        "gender": "M",
        "language": "English",
        "subscription_tier": "Premium",
        "account_age_days": 730,
        "preferred_genres": ["Sci-Fi", "Drama", "Thriller"],
    }
    
    interactions = [
        {"title": "Stranger Things", "type": "series", "genre": "Sci-Fi", "user_rating": 5, "watch_percentage": 95, "timestamp": "2024-11-01"},
        {"title": "Dark", "type": "series", "genre": "Sci-Fi", "user_rating": 5, "watch_percentage": 100, "timestamp": "2024-11-05"},
        {"title": "Inception", "type": "movie", "genre": "Sci-Fi", "user_rating": 4, "watch_percentage": 90, "timestamp": "2024-11-10"},
        {"title": "Breaking Bad", "type": "series", "genre": "Drama", "user_rating": 5, "watch_percentage": 100, "timestamp": "2024-11-15"},
        {"title": "Interstellar", "type": "movie", "genre": "Sci-Fi", "user_rating": 5, "watch_percentage": 95, "timestamp": "2024-11-20"},
    ]
    
    context = {
        "time_of_day": "evening",
        "day_of_week": "Friday",
        "device": "Smart TV",
        "location_type": "home",
        "session_duration_minutes": 45,
        "items_viewed": 8,
        "items_clicked": 3,
        "recent_searches": ["sci-fi series", "time travel movies"],
    }
    
    return catalog, user_data, interactions, context


@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')


@app.route('/recommend', methods=['POST'])
def recommend():
    """Generate recommendations"""
    data = request.get_json()
    
    user_data = data.get('user_data', {})
    interactions = data.get('interactions', [])
    candidates = data.get('candidates', [])
    context = data.get('context', {})
    
    result = server.recommend(
        user_data=user_data,
        interactions=interactions,
        candidates=candidates,
        context=context,
        verbalizer=verbalizer,
        context_engine=context_engine,
    )
    
    return jsonify({
        'recommendations': result.recommendations,
        'scores': result.scores,
        'latency_ms': result.latency_ms,
        'cache_hit': result.cache_hit,
        'metadata': result.metadata,
    })


@app.route('/demo')
def demo():
    """Run a demo recommendation"""
    catalog, user_data, interactions, context = load_sample_data()
    
    result = server.recommend(
        user_data=user_data,
        interactions=interactions,
        candidates=catalog,
        context=context,
        verbalizer=verbalizer,
        context_engine=context_engine,
    )
    
    # Get full item details for recommendations
    rec_items = []
    for item_id in result.recommendations:
        item = next((c for c in catalog if c['id'] == item_id), None)
        if item:
            rec_items.append(item)
    
    return render_template('results.html',
                         recommendations=rec_items,
                         scores=result.scores,
                         latency_ms=result.latency_ms,
                         cache_hit=result.cache_hit,
                         user_data=user_data,
                         interactions=interactions)


@app.route('/api-docs')
def api_docs():
    """API documentation page"""
    return render_template('api_docs.html')


@app.route('/api/health')
def health():
    """Health check endpoint"""
    return jsonify(server.health_check())


@app.route('/api/metrics')
def metrics():
    """Get serving metrics"""
    return jsonify(server.get_metrics())


@app.route('/api/evaluate', methods=['POST'])
def evaluate():
    """Evaluate rankings"""
    data = request.get_json()
    rankings = data.get('rankings', [])
    relevance_grades = data.get('relevance_grades', [])
    
    metrics = evaluator.evaluate_offline(rankings, relevance_grades)
    
    return jsonify({
        'ndcg': metrics.ndcg,
        'mrr': metrics.mrr,
        'map': metrics.map,
        'precision': metrics.precision,
        'recall': metrics.recall,
        'coverage': metrics.coverage,
        'diversity': metrics.diversity,
        'novelty': metrics.novelty,
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
