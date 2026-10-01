"""
GenRec Demo Script
Quick demonstration of the LLM-Backed Recommendation Ranker

Based on: "GenRec: An LLM-Backed Recommendation Ranker at Netflix"
Built by: Sandesh Dhital
"""

import json
from genrec.verbalizer import Verbalizer
from genrec.context_engine import ContextEngine
from genrec.ranker import GenRecRanker
from genrec.reward import RewardModel
from genrec.serving import GenRecServer, ServingConfig
from genrec.evaluation import Evaluator


def main():
    print("=" * 70)
    print("  GenRec: LLM-Backed Recommendation Ranker - Demo")
    print("  Based on: 'GenRec: An LLM-Backed Recommendation Ranker at Netflix'")
    print("  Built by: Sandesh Dhital")
    print("=" * 70)
    print()
    
    # Initialize components
    print("[1/5] Initializing components...")
    verbalizer = Verbalizer(max_history=10)
    context_engine = ContextEngine()
    reward_model = RewardModel()
    ranker = GenRecRanker(use_openai=False)  # Use mock ranker for demo
    server = GenRecServer(ranker, ServingConfig(max_candidates=15, top_k=5))
    evaluator = Evaluator(k_values=[1, 3, 5, 10])
    print("  Components initialized successfully!")
    print()
    
    # Load sample data
    print("[2/5] Loading sample data...")
    catalog = [
        {"id": "item_1", "title": "Stranger Things", "type": "series", "genres": ["Sci-Fi", "Horror", "Drama"], "year": 2016, "rating": 8.7, "duration_minutes": 50, "description": "A group of kids uncover supernatural mysteries in their small town."},
        {"id": "item_2", "title": "The Crown", "type": "series", "genres": ["Drama", "History", "Biography"], "year": 2016, "rating": 8.6, "duration_minutes": 60, "description": "The reign of Queen Elizabeth II from the 1940s to modern times."},
        {"id": "item_3", "title": "Extraction", "type": "movie", "genres": ["Action", "Thriller"], "year": 2020, "rating": 6.7, "duration_minutes": 117, "description": "A black-market mercenary hired to rescue the kidnapped son of a crime lord."},
        {"id": "item_4", "title": "The Witcher", "type": "series", "genres": ["Fantasy", "Action", "Adventure"], "year": 2019, "rating": 8.2, "duration_minutes": 60, "description": "Geralt of Rivia, a solitary monster hunter, struggles to find his place in a dangerous world."},
        {"id": "item_5", "title": "Money Heist", "type": "series", "genres": ["Crime", "Drama", "Thriller"], "year": 2017, "rating": 8.2, "duration_minutes": 70, "description": "An unusual group of robbers attempt to carry out the most perfect robbery in Spanish history."},
        {"id": "item_6", "title": "Bird Box", "type": "movie", "genres": ["Horror", "Thriller", "Drama"], "year": 2018, "rating": 6.6, "duration_minutes": 124, "description": "A mother and her two children make a desperate bid to reach safety."},
        {"id": "item_7", "title": "Dark", "type": "series", "genres": ["Sci-Fi", "Drama", "Crime"], "year": 2017, "rating": 8.8, "duration_minutes": 53, "description": "A family saga with a supernatural twist in a German town."},
        {"id": "item_8", "title": "The Queen's Gambit", "type": "series", "genres": ["Drama"], "year": 2020, "rating": 8.5, "duration_minutes": 375, "description": "The story of an orphaned chess prodigy's rise to the top."},
        {"id": "item_9", "title": "Inception", "type": "movie", "genres": ["Sci-Fi", "Action", "Thriller"], "year": 2010, "rating": 8.8, "duration_minutes": 148, "description": "A thief who steals secrets through dream-sharing technology."},
        {"id": "item_10", "title": "Breaking Bad", "type": "series", "genres": ["Crime", "Drama", "Thriller"], "year": 2008, "rating": 9.5, "duration_minutes": 49, "description": "A high school chemistry teacher turned methamphetamine producer."},
        {"id": "item_11", "title": "Interstellar", "type": "movie", "genres": ["Sci-Fi", "Adventure", "Drama"], "year": 2014, "rating": 8.7, "duration_minutes": 169, "description": "A team of explorers travel through a wormhole in space."},
        {"id": "item_12", "title": "The Office", "type": "series", "genres": ["Comedy"], "year": 2005, "rating": 9.0, "duration_minutes": 22, "description": "A mockumentary on a group of typical office workers."},
        {"id": "item_13", "title": "Squid Game", "type": "series", "genres": ["Drama", "Thriller", "Action"], "year": 2021, "rating": 8.0, "duration_minutes": 55, "description": "Hundreds of cash-strapped players compete in children's games for a prize."},
        {"id": "item_14", "title": "Dune", "type": "movie", "genres": ["Sci-Fi", "Adventure"], "year": 2021, "rating": 8.0, "duration_minutes": 155, "description": "Feature adaptation of Frank Herbert's science fiction novel."},
        {"id": "item_15", "title": "Wednesday", "type": "series", "genres": ["Comedy", "Fantasy", "Mystery"], "year": 2022, "rating": 8.5, "duration_minutes": 45, "description": "Follows Wednesday Addams' years as a student at Nevermore Academy."},
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
    print("  Sample data loaded: 15 items, 5 interactions")
    print()
    
    # Demonstrate verbalization
    print("[3/5] Demonstrating input verbalization...")
    engineered_context = context_engine.engineer_context(context)
    prompt = verbalizer.create_ranking_prompt(user_data, interactions, catalog, engineered_context)
    print(f"  Generated prompt length: {len(prompt)} characters")
    print(f"  Context engineered: {context_engine.to_verbalized_string(engineered_context)}")
    print()
    
    # Generate recommendations
    print("[4/5] Generating recommendations...")
    result = server.recommend(
        user_data=user_data,
        interactions=interactions,
        candidates=catalog,
        context=context,
        verbalizer=verbalizer,
        context_engine=context_engine,
    )
    
    print(f"  Latency: {result.latency_ms:.2f}ms")
    print(f"  Cache hit: {result.cache_hit}")
    print()
    print("  Top 5 Recommendations:")
    print("  " + "-" * 50)
    for i, item_id in enumerate(result.recommendations[:5], 1):
        item = next((c for c in catalog if c['id'] == item_id), None)
        if item:
            score = result.scores.get(item_id, 0)
            print(f"  {i}. {item['title']} ({item['year']}) - Score: {score:.4f}")
            print(f"     Genres: {', '.join(item['genres'])}")
            print(f"     {item['description'][:60]}...")
    print()
    
    # Demonstrate reward model
    print("[5/5] Computing reward signals...")
    interactions_dict = {item['id']: {"clicked": True, "watch_time_minutes": 45, "duration_minutes": 50, "completed": True, "user_rating": 5} for item in catalog[:5]}
    rewards = reward_model.compute_total_reward(
        result.recommendations,
        interactions_dict,
        item_features={c['id']: c for c in catalog},
        user_history=[i['title'] for i in interactions],
    )
    print(f"  Ranking reward: {rewards.get('ranking', 0):.4f}")
    print(f"  Diversity reward: {rewards.get('diversity', 0):.4f}")
    print(f"  Novelty reward: {rewards.get('novelty', 0):.4f}")
    print(f"  Total reward: {rewards.get('total', 0):.4f}")
    print()
    
    # Show metrics
    print("=" * 70)
    print("  Serving Metrics:")
    metrics = server.get_metrics()
    for key, value in metrics.items():
        print(f"    {key}: {value}")
    print("=" * 70)
    print()
    print("Demo complete! Run 'python app.py' to start the web interface.")


if __name__ == '__main__':
    main()
