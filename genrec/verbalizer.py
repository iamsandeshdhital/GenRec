"""
GenRec Input Verbalization Module

Converts user histories and item features into natural language text
for LLM consumption, replacing traditional feature engineering.

Based on: "GenRec: An LLM-Backed Recommendation Ranker at Netflix"
Section: Input Verbalization and Context Engineering

Built by: Sandesh Dhital
"""

import json
from typing import List, Dict, Any, Optional
from datetime import datetime


class Verbalizer:
    """
    Converts structured user interaction data into natural language
    verbalizations that can be consumed by an LLM ranker.
    """
    
    DEFAULT_TEMPLATE = """You are a recommendation system. Based on the user's viewing history and preferences, rank the following candidate items by relevance.

## User Profile
{user_profile}

## Viewing History
{viewing_history}

## Current Context
{context}

## Candidate Items
{candidates}

Please rank the candidates from most to least relevant. Provide the ranking as a numbered list with brief explanations.
"""
    
    def __init__(self, template: Optional[str] = None, max_history: int = 20):
        self.template = template or self.DEFAULT_TEMPLATE
        self.max_history = max_history
    
    def verbalize_user_profile(self, user_data: Dict[str, Any]) -> str:
        """Convert user profile data to natural language"""
        parts = []
        
        if 'age_group' in user_data:
            parts.append(f"Age group: {user_data['age_group']}")
        if 'gender' in user_data:
            parts.append(f"Gender: {user_data['gender']}")
        if 'language' in user_data:
            parts.append(f"Preferred language: {user_data['language']}")
        if 'subscription_tier' in user_data:
            parts.append(f"Subscription: {user_data['subscription_tier']}")
        if 'account_age_days' in user_data:
            parts.append(f"Member for: {user_data['account_age_days']} days")
        if 'preferred_genres' in user_data:
            genres = ', '.join(user_data['preferred_genres'])
            parts.append(f"Preferred genres: {genres}")
        
        return '\n'.join(parts) if parts else 'No profile information available.'
    
    def verbalize_history(self, interactions: List[Dict[str, Any]]) -> str:
        """Convert viewing history to natural language"""
        if not interactions:
            return 'No viewing history available.'
        
        parts = []
        for i, interaction in enumerate(interactions[-self.max_history:], 1):
            item = interaction.get('title', 'Unknown')
            item_type = interaction.get('type', 'title')
            genre = interaction.get('genre', 'unknown genre')
            rating = interaction.get('user_rating')
            watch_pct = interaction.get('watch_percentage', 0)
            timestamp = interaction.get('timestamp', '')
            
            entry = f"{i}. Watched '{item}' ({item_type}, {genre})"
            
            if watch_pct:
                entry += f" - watched {watch_pct:.0f}%"
            if rating:
                entry += f" - rated {rating}/5"
            if timestamp:
                entry += f" on {timestamp}"
            
            parts.append(entry)
        
        return '\n'.join(parts)
    
    def verbalize_candidates(self, candidates: List[Dict[str, Any]]) -> str:
        """Convert candidate items to natural language"""
        if not candidates:
            return 'No candidate items provided.'
        
        parts = []
        for i, candidate in enumerate(candidates, 1):
            title = candidate.get('title', 'Unknown')
            item_type = candidate.get('type', 'title')
            genre = candidate.get('genre', 'unknown genre')
            year = candidate.get('year', '')
            rating = candidate.get('rating', '')
            description = candidate.get('description', '')
            duration = candidate.get('duration_minutes', '')
            
            entry = f"{i}. '{title}'"
            if year:
                entry += f" ({year})"
            entry += f" - {item_type}, {genre}"
            if rating:
                entry += f", rated {rating}/10"
            if duration:
                entry += f", {duration} min"
            if description:
                entry += f"\n   Description: {description}"
            
            parts.append(entry)
        
        return '\n'.join(parts)
    
    def verbalize_context(self, context: Dict[str, Any]) -> str:
        """Convert contextual signals to natural language"""
        parts = []
        
        if 'time_of_day' in context:
            parts.append(f"Time: {context['time_of_day']}")
        if 'day_of_week' in context:
            parts.append(f"Day: {context['day_of_week']}")
        if 'device' in context:
            parts.append(f"Device: {context['device']}")
        if 'location' in context:
            parts.append(f"Location: {context['location']}")
        if 'session_duration_minutes' in context:
            parts.append(f"Session duration: {context['session_duration_minutes']} minutes")
        if 'recent_searches' in context:
            searches = ', '.join(context['recent_searches'])
            parts.append(f"Recent searches: {searches}")
        if 'current_mood' in context:
            parts.append(f"Current mood: {context['current_mood']}")
        
        return '\n'.join(parts) if parts else 'No contextual information available.'
    
    def create_prompt(self, user_data: Dict[str, Any], interactions: List[Dict[str, Any]],
                      candidates: List[Dict[str, Any]], context: Dict[str, Any]) -> str:
        """Create the full verbalized prompt for the LLM"""
        return self.template.format(
            user_profile=self.verbalize_user_profile(user_data),
            viewing_history=self.verbalize_history(interactions),
            context=self.verbalize_context(context),
            candidates=self.verbalize_candidates(candidates)
        )
    
    def create_ranking_prompt(self, user_data: Dict[str, Any], interactions: List[Dict[str, Any]],
                              candidates: List[Dict[str, Any]], context: Dict[str, Any]) -> str:
        """Create a focused ranking prompt"""
        prompt = f"""Rank the following {len(candidates)} items for this user based on their history and context.

User has watched: {self._summarize_history(interactions)}

Current context: {self._summarize_context(context)}

Candidates:
{self.verbalize_candidates(candidates)}

Output format: Provide a JSON array of item IDs in ranked order (most relevant first).
Example: ["item_3", "item_1", "item_2"]
"""
        return prompt
    
    def _summarize_history(self, interactions: List[Dict[str, Any]]) -> str:
        """Create a brief summary of viewing history"""
        if not interactions:
            return "No previous viewing history"
        
        genres = {}
        recent_titles = []
        for interaction in interactions[-10:]:
            genre = interaction.get('genre', 'unknown')
            genres[genre] = genres.get(genre, 0) + 1
            recent_titles.append(interaction.get('title', 'Unknown'))
        
        top_genres = sorted(genres.items(), key=lambda x: x[1], reverse=True)[:3]
        genre_str = ', '.join([f"{g} ({c})" for g, c in top_genres])
        
        return f"Top genres: {genre_str}. Recently watched: {', '.join(recent_titles[-5:])}"
    
    def _summarize_context(self, context: Dict[str, Any]) -> str:
        """Create a brief summary of context"""
        parts = []
        if 'time_of_day' in context:
            parts.append(context['time_of_day'])
        if 'device' in context:
            parts.append(f"on {context['device']}")
        if 'day_of_week' in context:
            parts.append(context['day_of_week'])
        return ' '.join(parts) if parts else "No specific context"
