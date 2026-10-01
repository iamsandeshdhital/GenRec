"""
GenRec Evaluation Module

Implements offline and online evaluation metrics for the
LLM-backed recommendation ranker.

Based on: "GenRec: An LLM-Backed Recommendation Ranker at Netflix"
Section: Evaluation

Built by: Sandesh Dhital
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics"""
    ndcg: float
    mrr: float
    precision: Dict[int, float]
    recall: Dict[int, float]
    map: float
    mrr_at_k: Dict[int, float]
    hit_rate: Dict[int, float]
    coverage: float
    diversity: float
    novelty: float


class Evaluator:
    """
    Evaluates GenRec ranking performance.
    
    Implements both offline metrics (NDCG, MRR, Precision, Recall, MAP)
    and online metrics (CTR, watch time, retention).
    """
    
    def __init__(self, k_values: List[int] = [5, 10, 20, 50]):
        self.k_values = k_values
    
    def evaluate_offline(self, rankings: List[List[str]],
                         relevance_grades: List[Dict[str, float]]) -> EvaluationMetrics:
        """
        Evaluate offline ranking performance.
        
        Args:
            rankings: List of ranked item ID lists
            relevance_grades: List of dicts mapping item IDs to relevance grades
            
        Returns:
            EvaluationMetrics with all computed metrics
        """
        ndcg_scores = []
        mrr_scores = []
        precision_at_k = defaultdict(list)
        recall_at_k = defaultdict(list)
        ap_scores = []
        hit_rate_at_k = defaultdict(list)
        
        for ranking, grades in zip(rankings, relevance_grades):
            # NDCG
            ndcg = self._compute_ndcg(ranking, grades)
            ndcg_scores.append(ndcg)
            
            # MRR
            mrr = self._compute_mrr(ranking, grades)
            mrr_scores.append(mrr)
            
            # Precision and Recall at k
            relevant_items = {item for item, grade in grades.items() if grade > 0}
            
            for k in self.k_values:
                top_k = ranking[:k]
                
                # Precision@k
                precision = len(set(top_k) & relevant_items) / k if k > 0 else 0
                precision_at_k[k].append(precision)
                
                # Recall@k
                recall = len(set(top_k) & relevant_items) / len(relevant_items) if relevant_items else 0
                recall_at_k[k].append(recall)
                
                # Hit Rate@k
                hit = 1 if len(set(top_k) & relevant_items) > 0 else 0
                hit_rate_at_k[k].append(hit)
            
            # Average Precision
            ap = self._compute_ap(ranking, grades)
            ap_scores.append(ap)
        
        # Compute coverage
        all_items = set()
        for ranking in rankings:
            all_items.update(ranking)
        coverage = len(all_items) / 1000  # Assuming catalog size of 1000
        
        # Compute diversity
        diversity = self._compute_diversity(rankings)
        
        # Compute novelty
        novelty = self._compute_novelty(rankings)
        
        return EvaluationMetrics(
            ndcg=np.mean(ndcg_scores),
            mrr=np.mean(mrr_scores),
            precision={k: np.mean(v) for k, v in precision_at_k.items()},
            recall={k: np.mean(v) for k, v in recall_at_k.items()},
            map=np.mean(ap_scores),
            mrr_at_k={k: np.mean(v) for k, v in hit_rate_at_k.items()},
            hit_rate={k: np.mean(v) for k, v in hit_rate_at_k.items()},
            coverage=coverage,
            diversity=diversity,
            novelty=novelty,
        )
    
    def evaluate_online(self, experiment_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate online A/B test performance.
        
        Args:
            experiment_data: A/B test data with control and treatment groups
            
        Returns:
            Dictionary of online metrics
        """
        control = experiment_data.get('control', {})
        treatment = experiment_data.get('treatment', {})
        
        metrics = {}
        
        # CTR
        metrics['ctr_lift'] = self._compute_lift(
            control.get('ctr', 0), treatment.get('ctr', 0)
        )
        
        # Watch time
        metrics['watch_time_lift'] = self._compute_lift(
            control.get('watch_time', 0), treatment.get('watch_time', 0)
        )
        
        # Retention
        metrics['retention_lift'] = self._compute_lift(
            control.get('retention', 0), treatment.get('retention', 0)
        )
        
        # Satisfaction
        metrics['satisfaction_lift'] = self._compute_lift(
            control.get('satisfaction', 0), treatment.get('satisfaction', 0)
        )
        
        return metrics
    
    def _compute_ndcg(self, ranking: List[str], grades: Dict[str, float]) -> float:
        """Compute Normalized Discounted Cumulative Gain"""
        dcg = 0.0
        for i, item in enumerate(ranking):
            relevance = grades.get(item, 0)
            dcg += relevance / np.log2(i + 2)
        
        # Ideal DCG
        ideal_relevances = sorted(grades.values(), reverse=True)
        idcg = 0.0
        for i, relevance in enumerate(ideal_relevances):
            idcg += relevance / np.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def _compute_mrr(self, ranking: List[str], grades: Dict[str, float]) -> float:
        """Compute Mean Reciprocal Rank"""
        for i, item in enumerate(ranking):
            if grades.get(item, 0) > 0:
                return 1.0 / (i + 1)
        return 0.0
    
    def _compute_ap(self, ranking: List[str], grades: Dict[str, float]) -> float:
        """Compute Average Precision"""
        relevant_items = {item for item, grade in grades.items() if grade > 0}
        
        if not relevant_items:
            return 0.0
        
        precision_sum = 0.0
        relevant_count = 0
        
        for i, item in enumerate(ranking):
            if item in relevant_items:
                relevant_count += 1
                precision_sum += relevant_count / (i + 1)
        
        return precision_sum / len(relevant_items)
    
    def _compute_diversity(self, rankings: List[List[str]]) -> float:
        """Compute intra-list diversity"""
        diversities = []
        
        for ranking in rankings:
            if len(ranking) < 2:
                continue
            
            # Compute pairwise diversity (simplified)
            unique_items = len(set(ranking))
            diversity = unique_items / len(ranking)
            diversities.append(diversity)
        
        return np.mean(diversities) if diversities else 0.0
    
    def _compute_novelty(self, rankings: List[List[str]]) -> float:
        """Compute novelty (inverse popularity)"""
        # Count item frequencies
        item_counts = defaultdict(int)
        total_items = 0
        
        for ranking in rankings:
            for item in ranking:
                item_counts[item] += 1
                total_items += 1
        
        if not item_counts:
            return 0.0
        
        # Compute average novelty
        novelties = []
        for ranking in rankings:
            for item in ranking:
                popularity = item_counts[item] / total_items
                novelty = -np.log2(popularity) if popularity > 0 else 0
                novelties.append(novelty)
        
        return np.mean(novelties) if novelties else 0.0
    
    def _compute_lift(self, control: float, treatment: float) -> float:
        """Compute relative lift"""
        if control == 0:
            return 0.0
        return (treatment - control) / control
    
    def generate_report(self, offline_metrics: EvaluationMetrics,
                        online_metrics: Optional[Dict[str, float]] = None) -> str:
        """Generate a formatted evaluation report"""
        report = []
        report.append("=" * 60)
        report.append("GenRec Evaluation Report")
        report.append("=" * 60)
        report.append("")
        report.append("Offline Metrics:")
        report.append(f"  NDCG: {offline_metrics.ndcg:.4f}")
        report.append(f"  MRR: {offline_metrics.mrr:.4f}")
        report.append(f"  MAP: {offline_metrics.map:.4f}")
        report.append("")
        
        report.append("Precision@K:")
        for k, v in offline_metrics.precision.items():
            report.append(f"  P@{k}: {v:.4f}")
        
        report.append("")
        report.append("Recall@K:")
        for k, v in offline_metrics.recall.items():
            report.append(f"  R@{k}: {v:.4f}")
        
        report.append("")
        report.append(f"Coverage: {offline_metrics.coverage:.4f}")
        report.append(f"Diversity: {offline_metrics.diversity:.4f}")
        report.append(f"Novelty: {offline_metrics.novelty:.4f}")
        
        if online_metrics:
            report.append("")
            report.append("Online Metrics (A/B Test):")
            for metric, value in online_metrics.items():
                report.append(f"  {metric}: {value:+.2%}")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)
