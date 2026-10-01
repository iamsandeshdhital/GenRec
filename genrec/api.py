"""
GenRec API Module
RESTful API endpoints for the recommendation system

Built by: Sandesh Dhital
"""

from flask import Blueprint, request, jsonify
from typing import Dict, Any, List
import time

from .verbalizer import Verbalizer
from .context_engine import ContextEngine
from .ranker import GenRecRanker
from .reward import RewardModel
from .serving import GenRecServer, ServingConfig
from .evaluation import Evaluator

api_bp = Blueprint('api', __name__, url_prefix='/api')

# Initialize components
verbalizer = Verbalizer(max_history=10)
context_engine = ContextEngine()
reward_model = RewardModel()
evaluator = Evaluator()

# Initialize ranker
import os
use_openai = bool(os.environ.get('OPENAI_API_KEY'))
ranker = GenRecRanker(use_openai=use_openai)

# Initialize server
server = GenRecServer(ranker, ServingConfig())


@api_bp.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify(server.health_check())


@api_bp.route('/metrics', methods=['GET'])
def metrics():
    """Get serving metrics"""
    return jsonify(server.get_metrics())


@api_bp.route('/recommend', methods=['POST'])
def recommend():
    """Generate recommendations"""
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    user_data = data.get('user_data', {})
    interactions = data.get('interactions', [])
    candidates = data.get('candidates', [])
    context = data.get('context', {})
    
    if not candidates:
        return jsonify({'error': 'No candidates provided'}), 400
    
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


@api_bp.route('/evaluate', methods=['POST'])
def evaluate():
    """Evaluate rankings"""
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    rankings = data.get('rankings', [])
    relevance_grades = data.get('relevance_grades', [])
    
    if not rankings or not relevance_grades:
        return jsonify({'error': 'Rankings and relevance grades required'}), 400
    
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


@api_bp.route('/reward', methods=['POST'])
def compute_reward():
    """Compute reward for a ranking"""
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    ranked_items = data.get('ranked_items', [])
    interactions = data.get('interactions', {})
    
    if not ranked_items:
        return jsonify({'error': 'Ranked items required'}), 400
    
    rewards = reward_model.compute_total_reward(
        ranked_items,
        interactions,
    )
    
    return jsonify(rewards)


@api_bp.route('/verbalize', methods=['POST'])
def verbalize():
    """Verbalize user data and candidates"""
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    user_data = data.get('user_data', {})
    interactions = data.get('interactions', [])
    candidates = data.get('candidates', [])
    context = data.get('context', {})
    
    engineered_context = context_engine.engineer_context(context)
    prompt = verbalizer.create_ranking_prompt(
        user_data, interactions, candidates, engineered_context
    )
    
    return jsonify({
        'prompt': prompt,
        'context_summary': context_engine.to_verbalized_string(engineered_context),
    })


@api_bp.route('/cache/clear', methods=['POST'])
def clear_cache():
    """Clear the prefill cache"""
    server.clear_cache()
    return jsonify({'message': 'Cache cleared successfully'})
