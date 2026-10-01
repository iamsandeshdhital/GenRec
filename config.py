"""
GenRec Configuration
Based on: "GenRec: An LLM-Backed Recommendation Ranker at Netflix"
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """GenRec configuration"""
    
    # LLM Configuration
    LLM_MODEL = os.environ.get('LLM_MODEL', 'meta-llama/Llama-2-7b-chat-hf')
    LLM_TEMPERATURE = float(os.environ.get('LLM_TEMPERATURE', '0.7'))
    LLM_MAX_TOKENS = int(os.environ.get('LLM_MAX_TOKENS', '512'))
    LLM_TOP_P = float(os.environ.get('LLM_TOP_P', '0.9'))
    
    # OpenAI Configuration (alternative to local LLM)
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')
    OPENAI_MODEL = os.environ.get('OPENAI_MODEL', 'gpt-4')
    
    # Training Configuration
    BATCH_SIZE = int(os.environ.get('BATCH_SIZE', '32'))
    LEARNING_RATE = float(os.environ.get('LEARNING_RATE', '2e-5'))
    NUM_EPOCHS = int(os.environ.get('NUM_EPOCHS', '3'))
    WARMUP_STEPS = int(os.environ.get('WARMUP_STEPS', '500'))
    MAX_GRAD_NORM = float(os.environ.get('MAX_GRAD_NORM', '1.0'))
    
    # Serving Configuration
    MAX_CANDIDATES = int(os.environ.get('MAX_CANDIDATES', '100'))
    TOP_K_RECOMMENDATIONS = int(os.environ.get('TOP_K_RECOMMENDATIONS', '10'))
    PREFILL_BATCH_SIZE = int(os.environ.get('PREFILL_BATCH_SIZE', '16'))
    
    # Verbalization Configuration
    MAX_HISTORY_LENGTH = int(os.environ.get('MAX_HISTORY_LENGTH', '20'))
    MAX_CONTEXT_LENGTH = int(os.environ.get('MAX_CONTEXT_LENGTH', '5'))
    VERBALIZATION_TEMPLATE = os.environ.get('VERBALIZATION_TEMPLATE', 'default')
    
    # Reward Configuration
    REWARD_WEIGHTS = {
        'click': 1.0,
        'watch_time': 2.0,
        'completion': 3.0,
        'rating': 4.0,
        'long_term_satisfaction': 5.0,
    }
    
    # Evaluation Configuration
    EVAL_METRICS = ['ndcg', 'mrr', 'precision', 'recall', 'map']
    ONLINE_METRICS = ['ctr', 'watch_time', 'retention', 'satisfaction']
    
    # Paths
    DATA_DIR = os.environ.get('DATA_DIR', 'data')
    MODEL_DIR = os.environ.get('MODEL_DIR', 'models')
    OUTPUT_DIR = os.environ.get('OUTPUT_DIR', 'outputs')
    CACHE_DIR = os.environ.get('CACHE_DIR', 'cache')
    
    # Flask
    SECRET_KEY = os.environ.get('SECRET_KEY', 'genrec-secret-key')
    DEBUG = os.environ.get('DEBUG', 'True').lower() == 'true'
