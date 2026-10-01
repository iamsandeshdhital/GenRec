"""
GenRec Reward Integration Module

Integrates business requirements and long-term member satisfaction
as reward signals for training the LLM ranker.

Based on: "GenRec: An LLM-Backed Recommendation Ranker at Netflix"
Section: Reward Integration

Built by: Sandesh Dhital
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class RewardSignal:
    """A single reward signal"""
    signal_type: str
    value: float
    weight: float
    timestamp: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class RewardModel:
    """
    Computes reward signals for training the GenRec ranker.
    
    Combines multiple reward signals:
    - Click-through rate
    - Watch time
    - Completion rate
    - User ratings
    - Long-term satisfaction metrics
    """
    
    DEFAULT_WEIGHTS = {
        'click': 1.0,
        'watch_time': 2.0,
        'completion': 3.0,
        'rating': 4.0,
        'long_term_satisfaction': 5.0,
        'diversity': 1.5,
        'novelty': 1.0,
        'freshness': 0.5,
    }
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
    
    def compute_reward(self, interaction: Dict[str, Any]) -> float:
        """
        Compute reward for a single interaction.
        
        Args:
            interaction: Dictionary containing interaction data
            
        Returns:
            Computed reward value
        """
        reward = 0.0
        
        # Click reward
        if interaction.get('clicked', False):
            reward += self.weights['click']
        
        # Watch time reward (normalized)
        watch_time = interaction.get('watch_time_minutes', 0)
        expected_duration = interaction.get('duration_minutes', 1)
        watch_ratio = min(watch_time / expected_duration, 1.0)
        reward += self.weights['watch_time'] * watch_ratio
        
        # Completion reward
        if interaction.get('completed', False):
            reward += self.weights['completion']
        
        # Rating reward
        user_rating = interaction.get('user_rating')
        if user_rating:
            # Normalize rating to 0-1 scale
            normalized_rating = (user_rating - 1) / 4  # Assuming 1-5 scale
            reward += self.weights['rating'] * normalized_rating
        
        # Long-term satisfaction (if available)
        satisfaction = interaction.get('satisfaction_score')
        if satisfaction:
            reward += self.weights['long_term_satisfaction'] * satisfaction
        
        return reward
    
    def compute_ranking_reward(self, ranked_items: List[str],
                                interactions: Dict[str, Dict[str, Any]]) -> float:
        """
        Compute reward for a ranked list of items.
        
        Args:
            ranked_items: List of item IDs in ranked order
            interactions: Dictionary mapping item IDs to interaction data
            
        Returns:
            Computed ranking reward
        """
        total_reward = 0.0
        
        for position, item_id in enumerate(ranked_items):
            if item_id in interactions:
                interaction = interactions[item_id]
                base_reward = self.compute_reward(interaction)
                
                # Position discount (higher positions get more weight)
                position_discount = 1.0 / (1.0 + 0.1 * position)
                
                total_reward += base_reward * position_discount
        
        return total_reward
    
    def compute_diversity_reward(self, ranked_items: List[str],
                                  item_features: Dict[str, Dict[str, Any]]) -> float:
        """
        Compute diversity reward for a ranked list.
        
        Args:
            ranked_items: List of item IDs in ranked order
            item_features: Dictionary mapping item IDs to feature data
            
        Returns:
            Diversity reward value
        """
        if len(ranked_items) < 2:
            return 0.0
        
        # Compute pairwise diversity
        diversities = []
        for i in range(len(ranked_items) - 1):
            item1 = ranked_items[i]
            item2 = ranked_items[i + 1]
            
            if item1 in item_features and item2 in item_features:
                diversity = self._compute_item_diversity(
                    item_features[item1], item_features[item2]
                )
                diversities.append(diversity)
        
        if not diversities:
            return 0.0
        
        avg_diversity = np.mean(diversities)
        return self.weights['diversity'] * avg_diversity
    
    def compute_novelty_reward(self, ranked_items: List[str],
                                user_history: List[str]) -> float:
        """
        Compute novelty reward (how many items are new to the user).
        
        Args:
            ranked_items: List of item IDs in ranked order
            user_history: List of item IDs the user has already seen
            
        Returns:
            Novelty reward value
        """
        novel_items = [item for item in ranked_items if item not in user_history]
        novelty_ratio = len(novel_items) / len(ranked_items) if ranked_items else 0
        return self.weights['novelty'] * novelty_ratio
    
    def compute_freshness_reward(self, ranked_items: List[str],
                                  item_ages: Dict[str, int]) -> float:
        """
        Compute freshness reward (prefer newer content).
        
        Args:
            ranked_items: List of item IDs in ranked order
            item_ages: Dictionary mapping item IDs to age in days
            
        Returns:
            Freshness reward value
        """
        if not ranked_items:
            return 0.0
        
        freshness_scores = []
        for item_id in ranked_items:
            age = item_ages.get(item_id, 365)
            # Exponential decay based on age
            freshness = np.exp(-age / 30)  # 30-day half-life
            freshness_scores.append(freshness)
        
        avg_freshness = np.mean(freshness_scores)
        return self.weights['freshness'] * avg_freshness
    
    def compute_total_reward(self, ranked_items: List[str],
                              interactions: Dict[str, Dict[str, Any]],
                              item_features: Optional[Dict[str, Dict[str, Any]]] = None,
                              user_history: Optional[List[str]] = None,
                              item_ages: Optional[Dict[str, int]] = None) -> Dict[str, float]:
        """
        Compute total reward with all components.
        
        Returns:
            Dictionary with individual and total reward values
        """
        rewards = {
            'ranking': self.compute_ranking_reward(ranked_items, interactions),
        }
        
        if item_features:
            rewards['diversity'] = self.compute_diversity_reward(ranked_items, item_features)
        
        if user_history:
            rewards['novelty'] = self.compute_novelty_reward(ranked_items, user_history)
        
        if item_ages:
            rewards['freshness'] = self.compute_freshness_reward(ranked_items, item_ages)
        
        rewards['total'] = sum(rewards.values())
        return rewards
    
    def _compute_item_diversity(self, features1: Dict[str, Any],
                                 features2: Dict[str, Any]) -> float:
        """Compute diversity between two items based on features"""
        # Simple genre-based diversity
        genres1 = set(features1.get('genres', []))
        genres2 = set(features2.get('genres', []))
        
        if not genres1 or not genres2:
            return 0.5
        
        # Jaccard distance
        intersection = len(genres1 & genres2)
        union = len(genres1 | genres2)
        
        if union == 0:
            return 0.0
        
        return 1.0 - (intersection / union)
    
    def update_weights(self, new_weights: Dict[str, float]):
        """Update reward weights"""
        self.weights.update(new_weights)
    
    def get_weights(self) -> Dict[str, float]:
        """Get current reward weights"""
        return self.weights.copy()
