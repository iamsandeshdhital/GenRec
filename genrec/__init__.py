"""
GenRec: LLM-Backed Recommendation Ranker
Based on the paper by Netflix Research (2026)

Built by: Sandesh Dhital
"""

__version__ = '1.0.0'
__author__ = 'Sandesh Dhital'
__paper_authors__ = [
    'Ying Li', 'Shradha Sehgal', 'Arjun Rao', 'Rein Houthooft',
    'Yunan Hu', 'Yaochen Zhu', 'Sourabh Medapati', 'Yun Li',
    'Linas Baltrunas', 'Grace Huang', 'Ashish Rastogi', 'Kamelia Aryafar'
]

from .verbalizer import Verbalizer
from .context_engine import ContextEngine
from .ranker import GenRecRanker
from .reward import RewardModel
from .serving import GenRecServer
from .evaluation import Evaluator

__all__ = [
    'Verbalizer',
    'ContextEngine',
    'GenRecRanker',
    'RewardModel',
    'GenRecServer',
    'Evaluator',
]
