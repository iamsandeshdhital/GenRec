"""
GenRec Serving Module

Implements cost-constrained serving with prefill-only inference
for efficient real-time recommendations.

Based on: "GenRec: An LLM-Backed Recommendation Ranker at Netflix"
Section: Cost-Constrained Serving Design

Built by: Sandesh Dhital
"""

import time
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import hashlib


@dataclass
class ServingConfig:
    """Configuration for GenRec serving"""
    max_candidates: int = 100
    top_k: int = 10
    prefill_batch_size: int = 16
    cache_ttl_seconds: int = 300
    max_latency_ms: int = 500
    enable_prefill_cache: bool = True


@dataclass
class ServingResult:
    """Result from serving"""
    recommendations: List[str]
    scores: Dict[str, float]
    latency_ms: float
    cache_hit: bool
    metadata: Dict[str, Any]


class PrefillCache:
    """Cache for precomputed prefill results"""
    
    def __init__(self, ttl_seconds: int = 300):
        self.cache = {}
        self.ttl_seconds = ttl_seconds
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value if not expired"""
        if key in self.cache:
            value, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl_seconds:
                return value
            else:
                del self.cache[key]
        return None
    
    def set(self, key: str, value: Any):
        """Set cached value"""
        self.cache[key] = (value, time.time())
    
    def clear(self):
        """Clear all cached values"""
        self.cache.clear()
    
    def size(self) -> int:
        """Get cache size"""
        return len(self.cache)


class GenRecServer:
    """
    GenRec serving infrastructure.
    
    Implements cost-constrained serving with:
    - Prefill-only inference
    - Caching for repeated contexts
    - Batch processing
    - Latency monitoring
    """
    
    def __init__(self, ranker: Any, config: Optional[ServingConfig] = None):
        self.ranker = ranker
        self.config = config or ServingConfig()
        self.cache = PrefillCache(self.config.cache_ttl_seconds)
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.metrics = {
            'total_requests': 0,
            'cache_hits': 0,
            'total_latency_ms': 0,
        }
    
    def recommend(self, user_data: Dict[str, Any],
                  interactions: List[Dict[str, Any]],
                  candidates: List[Dict[str, Any]],
                  context: Dict[str, Any],
                  verbalizer: Any,
                  context_engine: Any) -> ServingResult:
        """
        Generate recommendations for a user.
        
        Args:
            user_data: User profile data
            interactions: User interaction history
            candidates: Candidate items to rank
            context: Current context
            verbalizer: Verbalizer instance
            context_engine: Context engine instance
            
        Returns:
            ServingResult with recommendations and metadata
        """
        start_time = time.time()
        
        # Check cache
        cache_key = self._compute_cache_key(user_data, interactions, candidates, context)
        cached_result = self.cache.get(cache_key) if self.config.enable_prefill_cache else None
        
        if cached_result:
            self.metrics['cache_hits'] += 1
            self.metrics['total_requests'] += 1
            return ServingResult(
                recommendations=cached_result['recommendations'],
                scores=cached_result['scores'],
                latency_ms=(time.time() - start_time) * 1000,
                cache_hit=True,
                metadata={'cached': True}
            )
        
        # Engineer context
        engineered_context = context_engine.engineer_context(context)
        
        # Verbalize prompt
        prompt = verbalizer.create_ranking_prompt(
            user_data, interactions, candidates, engineered_context
        )
        
        # Extract candidate IDs
        candidate_ids = [c.get('id', c.get('title', '')) for c in candidates]
        
        # Rank candidates
        ranking_result = self.ranker.rank(prompt, candidate_ids)
        
        # Get top-k
        top_k_items = ranking_result.ranked_items[:self.config.top_k]
        
        # Cache result
        if self.config.enable_prefill_cache:
            self.cache.set(cache_key, {
                'recommendations': top_k_items,
                'scores': ranking_result.scores,
            })
        
        latency = (time.time() - start_time) * 1000
        
        # Update metrics
        self.metrics['total_requests'] += 1
        self.metrics['total_latency_ms'] += latency
        
        return ServingResult(
            recommendations=top_k_items,
            scores=ranking_result.scores,
            latency_ms=latency,
            cache_hit=False,
            metadata={
                'cached': False,
                'candidates_considered': len(candidates),
                'ranking_latency_ms': ranking_result.latency_ms,
            }
        )
    
    def batch_recommend(self, requests: List[Dict[str, Any]],
                        verbalizer: Any,
                        context_engine: Any) -> List[ServingResult]:
        """
        Process multiple recommendation requests in batch.
        
        Args:
            requests: List of request dictionaries
            verbalizer: Verbalizer instance
            context_engine: Context engine instance
            
        Returns:
            List of ServingResult objects
        """
        results = []
        
        # Process in batches
        for i in range(0, len(requests), self.config.prefill_batch_size):
            batch = requests[i:i + self.config.prefill_batch_size]
            
            for request in batch:
                result = self.recommend(
                    user_data=request.get('user_data', {}),
                    interactions=request.get('interactions', []),
                    candidates=request.get('candidates', []),
                    context=request.get('context', {}),
                    verbalizer=verbalizer,
                    context_engine=context_engine,
                )
                results.append(result)
        
        return results
    
    def _compute_cache_key(self, user_data: Dict[str, Any],
                           interactions: List[Dict[str, Any]],
                           candidates: List[Dict[str, Any]],
                           context: Dict[str, Any]) -> str:
        """Compute cache key from request data"""
        # Create a hash of the request
        key_data = {
            'user_id': user_data.get('user_id', ''),
            'interactions_hash': hashlib.md5(
                json.dumps(interactions[-5:], sort_keys=True).encode()
            ).hexdigest(),
            'candidates_hash': hashlib.md5(
                json.dumps([c.get('id', '') for c in candidates], sort_keys=True).encode()
            ).hexdigest(),
            'context_hash': hashlib.md5(
                json.dumps(context, sort_keys=True).encode()
            ).hexdigest(),
        }
        return hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get serving metrics"""
        avg_latency = (
            self.metrics['total_latency_ms'] / self.metrics['total_requests']
            if self.metrics['total_requests'] > 0 else 0
        )
        cache_hit_rate = (
            self.metrics['cache_hits'] / self.metrics['total_requests']
            if self.metrics['total_requests'] > 0 else 0
        )
        
        return {
            'total_requests': self.metrics['total_requests'],
            'cache_hits': self.metrics['cache_hits'],
            'cache_hit_rate': cache_hit_rate,
            'average_latency_ms': avg_latency,
            'cache_size': self.cache.size(),
        }
    
    def clear_cache(self):
        """Clear the prefill cache"""
        self.cache.clear()
    
    def health_check(self) -> Dict[str, Any]:
        """Health check endpoint"""
        return {
            'status': 'healthy',
            'cache_size': self.cache.size(),
            'metrics': self.get_metrics(),
        }
