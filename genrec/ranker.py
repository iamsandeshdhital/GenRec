"""
GenRec LLM-Backed Ranking Module

Core ranking model that uses an LLM to rank candidate items
based on verbalized user history and context.

Based on: "GenRec: An LLM-Backed Recommendation Ranker at Netflix"
Section: Model Architecture

Built by: Sandesh Dhital
"""

import json
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


@dataclass
class RankingResult:
    """Result of a ranking operation"""
    ranked_items: List[str]
    scores: Dict[str, float]
    explanations: Dict[str, str]
    raw_response: str
    latency_ms: float


class GenRecRanker:
    """
    LLM-backed recommendation ranker.
    
    Uses a large language model to rank candidate items based on
    verbalized user history, context, and item descriptions.
    """
    
    def __init__(self, model_name: str = 'meta-llama/Llama-2-7b-chat-hf',
                 use_openai: bool = False, openai_model: str = 'gpt-4',
                 device: str = 'auto'):
        """
        Initialize the GenRec ranker.
        
        Args:
            model_name: HuggingFace model name for local inference
            use_openai: Whether to use OpenAI API instead of local model
            openai_model: OpenAI model name
            device: Device for local inference ('cpu', 'cuda', 'auto')
        """
        self.model_name = model_name
        self.use_openai = use_openai
        self.openai_model = openai_model
        self.device = device
        self.model = None
        self.tokenizer = None
        
        if not use_openai and TRANSFORMERS_AVAILABLE:
            self._load_local_model()
    
    def _load_local_model(self):
        """Load local HuggingFace model"""
        if not TRANSFORMERS_AVAILABLE:
            print("Warning: transformers not installed. Using mock ranker.")
            return
        
        print(f"Loading model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=self.device if self.device == 'auto' else None,
        )
        if self.device != 'auto':
            self.model = self.model.to(self.device)
        print("Model loaded successfully")
    
    def rank(self, prompt: str, candidate_ids: List[str],
             temperature: float = 0.7, max_tokens: int = 512) -> RankingResult:
        """
        Rank candidate items based on the verbalized prompt.
        
        Args:
            prompt: Verbalized prompt with user history and candidates
            candidate_ids: List of candidate item IDs
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            RankingResult with ranked items and scores
        """
        import time
        start_time = time.time()
        
        if self.use_openai and OPENAI_AVAILABLE:
            response = self._rank_openai(prompt, temperature, max_tokens)
        elif self.model is not None:
            response = self._rank_local(prompt, temperature, max_tokens)
        else:
            response = self._rank_mock(prompt, candidate_ids)
        
        latency = (time.time() - start_time) * 1000
        
        # Parse the response
        ranked_items, scores, explanations = self._parse_ranking_response(
            response, candidate_ids
        )
        
        return RankingResult(
            ranked_items=ranked_items,
            scores=scores,
            explanations=explanations,
            raw_response=response,
            latency_ms=latency,
        )
    
    def _rank_openai(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """Rank using OpenAI API"""
        response = openai.ChatCompletion.create(
            model=self.openai_model,
            messages=[
                {"role": "system", "content": "You are a recommendation ranking system. Always respond with a JSON array of item IDs in ranked order."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content
    
    def _rank_local(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """Rank using local HuggingFace model"""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract just the generated part
        response = response[len(prompt):]
        return response
    
    def _rank_mock(self, prompt: str, candidate_ids: List[str]) -> str:
        """Mock ranking for testing without a model"""
        # Simple heuristic: reverse the order as a mock ranking
        ranked = list(reversed(candidate_ids))
        return json.dumps(ranked)
    
    def _parse_ranking_response(self, response: str, candidate_ids: List[str]) -> Tuple[List[str], Dict[str, float], Dict[str, str]]:
        """Parse the LLM response into structured ranking"""
        ranked_items = []
        scores = {}
        explanations = {}
        
        # Try to parse JSON array
        try:
            # Find JSON array in response
            json_match = re.search(r'\[.*?\]', response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                if isinstance(parsed, list):
                    ranked_items = [str(item) for item in parsed]
        except (json.JSONDecodeError, AttributeError):
            pass
        
        # Fallback: try to find item IDs in response
        if not ranked_items:
            for candidate_id in candidate_ids:
                if candidate_id in response:
                    ranked_items.append(candidate_id)
        
        # Ensure all candidates are included
        for cid in candidate_ids:
            if cid not in ranked_items:
                ranked_items.append(cid)
        
        # Compute scores based on position
        n = len(ranked_items)
        for i, item_id in enumerate(ranked_items):
            # Exponential decay scoring
            scores[item_id] = 1.0 * (0.95 ** i)
            explanations[item_id] = f"Ranked #{i+1} by LLM"
        
        return ranked_items, scores, explanations
    
    def batch_rank(self, prompts: List[str], candidate_ids_list: List[List[str]],
                   temperature: float = 0.7, max_tokens: int = 512) -> List[RankingResult]:
        """Rank multiple prompts in batch"""
        results = []
        for prompt, candidate_ids in zip(prompts, candidate_ids_list):
            result = self.rank(prompt, candidate_ids, temperature, max_tokens)
            results.append(result)
        return results
