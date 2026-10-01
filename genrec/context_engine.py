"""
GenRec Context Engineering Module

Engineers rich contextual signals for the LLM ranker, replacing
traditional feature engineering with natural language context.

Based on: "GenRec: An LLM-Backed Recommendation Ranker at Netflix"
Section: Context Engineering

Built by: Sandesh Dhital
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import json


class ContextEngine:
    """
    Engineers contextual signals for recommendation ranking.
    Converts raw context data into rich, verbalized context.
    """
    
    def __init__(self):
        self.context_weights = {
            'temporal': 0.25,
            'behavioral': 0.25,
            'environmental': 0.20,
            'session': 0.15,
            'social': 0.15,
        }
    
    def engineer_context(self, raw_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Engineer comprehensive context from raw signals.
        
        Args:
            raw_context: Raw contextual data from the user session
            
        Returns:
            Engineered context dictionary
        """
        engineered = {
            'temporal': self._engineer_temporal_context(raw_context),
            'behavioral': self._engineer_behavioral_context(raw_context),
            'environmental': self._engineer_environmental_context(raw_context),
            'session': self._engineer_session_context(raw_context),
            'social': self._engineer_social_context(raw_context),
        }
        
        # Add computed context features
        engineered['computed'] = self._compute_context_features(engineered)
        
        return engineered
    
    def _engineer_temporal_context(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Engineer time-based context"""
        now = datetime.now()
        
        temporal = {
            'time_of_day': self._get_time_of_day(now),
            'day_of_week': now.strftime('%A'),
            'is_weekend': now.weekday() >= 5,
            'hour': now.hour,
            'season': self._get_season(now),
            'is_holiday': self._is_holiday(now),
        }
        
        # Add time since last session if available
        if 'last_session_time' in raw:
            last = datetime.fromisoformat(raw['last_session_time'])
            hours_since = (now - last).total_seconds() / 3600
            temporal['hours_since_last_session'] = round(hours_since, 1)
        
        return temporal
    
    def _engineer_behavioral_context(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Engineer user behavior context"""
        behavioral = {
            'current_session_duration': raw.get('session_duration_minutes', 0),
            'items_viewed_this_session': raw.get('items_viewed', 0),
            'items_clicked_this_session': raw.get('items_clicked', 0),
            'search_queries': raw.get('recent_searches', []),
            'browse_depth': raw.get('browse_depth', 0),
            'scroll_patterns': raw.get('scroll_patterns', {}),
        }
        
        # Compute engagement level
        behavioral['engagement_level'] = self._compute_engagement(behavioral)
        
        return behavioral
    
    def _engineer_environmental_context(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Engineer environmental/device context"""
        environmental = {
            'device_type': raw.get('device', 'unknown'),
            'screen_size': raw.get('screen_size', 'unknown'),
            'network_type': raw.get('network_type', 'unknown'),
            'location_type': raw.get('location_type', 'unknown'),  # home, work, travel
            'language': raw.get('language', 'en'),
            'accessibility_mode': raw.get('accessibility_mode', False),
        }
        
        return environmental
    
    def _engineer_session_context(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Engineer session-level context"""
        session = {
            'session_id': raw.get('session_id', ''),
            'entry_point': raw.get('entry_point', 'unknown'),  # homepage, search, notification
            'referrer': raw.get('referrer', 'direct'),
            'ab_test_group': raw.get('ab_test_group', 'control'),
            'feature_flags': raw.get('feature_flags', []),
        }
        
        return session
    
    def _engineer_social_context(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Engineer social context signals"""
        social = {
            'shared_content_recently': raw.get('shared_content', []),
            'social_activity': raw.get('social_activity', {}),
            'trending_in_network': raw.get('trending_in_network', []),
        }
        
        return social
    
    def _compute_context_features(self, engineered: Dict[str, Any]) -> Dict[str, Any]:
        """Compute derived context features"""
        computed = {}
        
        # Time-based features
        temporal = engineered.get('temporal', {})
        computed['is_prime_time'] = temporal.get('hour', 0) in [19, 20, 21, 22]
        computed['is_late_night'] = temporal.get('hour', 0) in [0, 1, 2, 3, 4, 5]
        computed['is_morning'] = temporal.get('hour', 0) in [6, 7, 8, 9, 10, 11]
        
        # Behavioral features
        behavioral = engineered.get('behavioral', {})
        computed['is_browsing'] = behavioral.get('items_viewed_this_session', 0) > 5
        computed['is_searching'] = len(behavioral.get('search_queries', [])) > 0
        computed['is_highly_engaged'] = behavioral.get('engagement_level', 0) > 0.7
        
        return computed
    
    def _get_time_of_day(self, dt: datetime) -> str:
        """Classify time of day"""
        hour = dt.hour
        if 5 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 17:
            return 'afternoon'
        elif 17 <= hour < 21:
            return 'evening'
        else:
            return 'night'
    
    def _get_season(self, dt: datetime) -> str:
        """Get season from date"""
        month = dt.month
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        else:
            return 'fall'
    
    def _is_holiday(self, dt: datetime) -> bool:
        """Check if date is a holiday (simplified)"""
        # In production, use a proper holiday calendar
        us_holidays = [
            (1, 1),   # New Year
            (7, 4),   # Independence Day
            (12, 25), # Christmas
            (12, 31), # New Year's Eve
        ]
        return (dt.month, dt.day) in us_holidays
    
    def _compute_engagement(self, behavioral: Dict[str, Any]) -> float:
        """Compute engagement level score"""
        items_viewed = behavioral.get('items_viewed_this_session', 0)
        items_clicked = behavioral.get('items_clicked_this_session', 0)
        session_duration = behavioral.get('current_session_duration', 0)
        
        # Simple engagement score
        if items_viewed == 0:
            return 0.0
        
        click_rate = items_clicked / items_viewed
        duration_score = min(session_duration / 30, 1.0)  # Cap at 30 min
        
        engagement = (click_rate * 0.6) + (duration_score * 0.4)
        return min(engagement, 1.0)
    
    def to_verbalized_string(self, engineered: Dict[str, Any]) -> str:
        """Convert engineered context to verbalized string"""
        parts = []
        
        temporal = engineered.get('temporal', {})
        if temporal:
            parts.append(f"Time: {temporal.get('time_of_day', 'unknown')}, {temporal.get('day_of_week', '')}")
        
        behavioral = engineered.get('behavioral', {})
        if behavioral:
            parts.append(f"Engagement: {behavioral.get('engagement_level', 0):.0%}")
            if behavioral.get('search_queries'):
                parts.append(f"Recent searches: {', '.join(behavioral['search_queries'][:3])}")
        
        environmental = engineered.get('environmental', {})
        if environmental:
            parts.append(f"Device: {environmental.get('device_type', 'unknown')}")
        
        computed = engineered.get('computed', {})
        if computed.get('is_prime_time'):
            parts.append("Prime time viewing")
        if computed.get('is_late_night'):
            parts.append("Late night session")
        
        return ' | '.join(parts) if parts else 'No context available'
