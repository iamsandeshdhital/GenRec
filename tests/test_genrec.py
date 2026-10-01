"""
Unit tests for GenRec modules
Built by: Sandesh Dhital
"""

import unittest
import json
from genrec.verbalizer import Verbalizer
from genrec.context_engine import ContextEngine
from genrec.reward import RewardModel
from genrec.evaluation import Evaluator


class TestVerbalizer(unittest.TestCase):
    def setUp(self):
        self.verbalizer = Verbalizer(max_history=5)
    
    def test_verbalize_user_profile(self):
        user_data = {
            "age_group": "25-34",
            "gender": "M",
            "language": "English",
            "preferred_genres": ["Sci-Fi", "Drama"],
        }
        result = self.verbalizer.verbalize_user_profile(user_data)
        self.assertIn("25-34", result)
        self.assertIn("Sci-Fi", result)
    
    def test_verbalize_history(self):
        interactions = [
            {"title": "Stranger Things", "genre": "Sci-Fi", "user_rating": 5, "watch_percentage": 95},
            {"title": "Dark", "genre": "Sci-Fi", "user_rating": 5, "watch_percentage": 100},
        ]
        result = self.verbalizer.verbalize_history(interactions)
        self.assertIn("Stranger Things", result)
        self.assertIn("Dark", result)
    
    def test_verbalize_candidates(self):
        candidates = [
            {"id": "item_1", "title": "Inception", "type": "movie", "genres": ["Sci-Fi"], "year": 2010, "rating": 8.8},
        ]
        result = self.verbalizer.verbalize_candidates(candidates)
        self.assertIn("Inception", result)
        self.assertIn("Sci-Fi", result)
    
    def test_create_prompt(self):
        user_data = {"age_group": "25-34"}
        interactions = [{"title": "Test", "genre": "Drama"}]
        candidates = [{"id": "item_1", "title": "Movie", "type": "movie", "genres": ["Action"]}]
        context = {"time_of_day": "evening"}
        
        result = self.verbalizer.create_prompt(user_data, interactions, candidates, context)
        self.assertIn("Test", result)
        self.assertIn("Movie", result)


class TestContextEngine(unittest.TestCase):
    def setUp(self):
        self.engine = ContextEngine()
    
    def test_engineer_context(self):
        raw_context = {
            "time_of_day": "evening",
            "day_of_week": "Friday",
            "device": "Smart TV",
            "session_duration_minutes": 45,
            "items_viewed": 8,
        }
        result = self.engine.engineer_context(raw_context)
        self.assertIn("temporal", result)
        self.assertIn("behavioral", result)
        self.assertIn("environmental", result)
    
    def test_time_of_day(self):
        from datetime import datetime
        dt = datetime(2024, 1, 1, 14, 0)  # 2 PM
        result = self.engine._get_time_of_day(dt)
        self.assertEqual(result, "afternoon")


class TestRewardModel(unittest.TestCase):
    def setUp(self):
        self.model = RewardModel()
    
    def test_compute_reward_click(self):
        interaction = {"clicked": True}
        reward = self.model.compute_reward(interaction)
        self.assertGreater(reward, 0)
    
    def test_compute_reward_no_click(self):
        interaction = {"clicked": False}
        reward = self.model.compute_reward(interaction)
        self.assertEqual(reward, 0)
    
    def test_compute_reward_with_rating(self):
        interaction = {"clicked": True, "user_rating": 5}
        reward = self.model.compute_reward(interaction)
        self.assertGreater(reward, 1)


class TestEvaluator(unittest.TestCase):
    def setUp(self):
        self.evaluator = Evaluator(k_values=[1, 3, 5])
    
    def test_compute_ndcg(self):
        ranking = ["item_1", "item_2", "item_3"]
        grades = {"item_1": 3, "item_2": 2, "item_3": 1}
        ndcg = self.evaluator._compute_ndcg(ranking, grades)
        self.assertGreater(ndcg, 0)
        self.assertLessEqual(ndcg, 1)
    
    def test_compute_mrr(self):
        ranking = ["item_1", "item_2", "item_3"]
        grades = {"item_2": 1}
        mrr = self.evaluator._compute_mrr(ranking, grades)
        self.assertEqual(mrr, 0.5)
    
    def test_evaluate_offline(self):
        rankings = [["item_1", "item_2", "item_3"]]
        relevance_grades = [{"item_1": 3, "item_2": 2, "item_3": 1}]
        metrics = self.evaluator.evaluate_offline(rankings, relevance_grades)
        self.assertGreater(metrics.ndcg, 0)
        self.assertGreater(metrics.mrr, 0)


if __name__ == '__main__':
    unittest.main()
