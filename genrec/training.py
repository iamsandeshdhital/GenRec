"""
GenRec Training Pipeline

Implements the two-phase training framework:
- Phase 1: Adapt LLM to domain data
- Phase 2: Post-train with recommendation-ranking specific data

Based on: "GenRec: An LLM-Backed Recommendation Ranker at Netflix"
Section: Post-Training Data Construction

Built by: Sandesh Dhital
"""

import json
import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
    )
    from datasets import Dataset
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


@dataclass
class TrainingExample:
    """A single training example"""
    prompt: str
    completion: str
    reward: float
    metadata: Optional[Dict[str, Any]] = None


class GenRecTrainer:
    """
    Training pipeline for GenRec.
    
    Implements Phase 1 (domain adaptation) and Phase 2 (ranking post-training).
    """
    
    def __init__(self, model_name: str = 'meta-llama/Llama-2-7b-chat-hf',
                 output_dir: str = 'outputs',
                 learning_rate: float = 2e-5,
                 num_epochs: int = 3,
                 batch_size: int = 4):
        self.model_name = model_name
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.model = None
        self.tokenizer = None
        
        if TRANSFORMERS_AVAILABLE:
            self._load_model()
    
    def _load_model(self):
        """Load the base model and tokenizer"""
        if not TRANSFORMERS_AVAILABLE:
            print("Warning: transformers not installed. Training disabled.")
            return
        
        print(f"Loading model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("Model loaded successfully")
    
    def prepare_phase1_data(self, catalog_data: List[Dict[str, Any]],
                            interaction_data: List[Dict[str, Any]]) -> List[TrainingExample]:
        """
        Prepare Phase 1 training data for domain adaptation.
        
        Phase 1 adapts the LLM to understand the content catalog
        and member behavior patterns.
        
        Args:
            catalog_data: Content catalog information
            interaction_data: User interaction data
            
        Returns:
            List of training examples
        """
        examples = []
        
        # Create catalog understanding examples
        for item in catalog_data[:100]:  # Limit for demo
            prompt = f"Describe the following content item:\nTitle: {item.get('title')}\nType: {item.get('type')}\nGenres: {', '.join(item.get('genres', []))}\nYear: {item.get('year')}"
            completion = f"This is a {item.get('type')} titled '{item.get('title')}' released in {item.get('year')}. It belongs to the {', '.join(item.get('genres', []))} genre(s). {item.get('description', '')}"
            examples.append(TrainingExample(prompt=prompt, completion=completion, reward=1.0))
        
        # Create user behavior understanding examples
        for interaction in interaction_data[:100]:
            prompt = f"User watched '{interaction.get('title')}' ({interaction.get('genre')}) and rated it {interaction.get('rating')}/5. What does this tell us about the user's preferences?"
            completion = f"The user enjoys {interaction.get('genre')} content and {'liked' if interaction.get('rating', 0) >= 4 else 'did not particularly enjoy'} this title."
            examples.append(TrainingExample(prompt=prompt, completion=completion, reward=1.0))
        
        return examples
    
    def prepare_phase2_data(self, ranking_data: List[Dict[str, Any]],
                            reward_model: Optional[Any] = None) -> List[TrainingExample]:
        """
        Prepare Phase 2 training data for ranking post-training.
        
        Phase 2 trains the model with recommendation-ranking specific
        data, labels, and reward signals.
        
        Args:
            ranking_data: Ranking training data with labels
            reward_model: Optional reward model for computing rewards
            
        Returns:
            List of training examples
        """
        examples = []
        
        for data_point in ranking_data:
            prompt = data_point.get('prompt', '')
            ideal_ranking = data_point.get('ideal_ranking', [])
            interactions = data_point.get('interactions', {})
            
            # Create completion from ideal ranking
            completion = json.dumps(ideal_ranking)
            
            # Compute reward if reward model provided
            reward = 1.0
            if reward_model:
                reward = reward_model.compute_ranking_reward(ideal_ranking, interactions)
            
            examples.append(TrainingExample(
                prompt=prompt,
                completion=completion,
                reward=reward,
                metadata=data_point.get('metadata', {})
            ))
        
        return examples
    
    def train_phase1(self, training_examples: List[TrainingExample],
                     output_dir: Optional[str] = None) -> str:
        """
        Execute Phase 1 training (domain adaptation).
        
        Args:
            training_examples: List of training examples
            output_dir: Output directory for the trained model
            
        Returns:
            Path to the trained model
        """
        if not TRANSFORMERS_AVAILABLE or self.model is None:
            print("Training not available. Skipping Phase 1.")
            return self.model_name
        
        output_dir = output_dir or os.path.join(self.output_dir, 'phase1')
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Starting Phase 1 training with {len(training_examples)} examples")
        
        # Convert to HuggingFace Dataset
        dataset = Dataset.from_list([
            {'text': ex.prompt + ex.completion} for ex in training_examples
        ])
        
        # Tokenize
        def tokenize_function(examples):
            return self.tokenizer(examples['text'], truncation=True, max_length=512)
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            warmup_steps=100,
            logging_steps=10,
            save_strategy='epoch',
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )
        
        # Train
        trainer.train()
        
        # Save model
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"Phase 1 training complete. Model saved to {output_dir}")
        return output_dir
    
    def train_phase2(self, training_examples: List[TrainingExample],
                     phase1_model_path: Optional[str] = None,
                     output_dir: Optional[str] = None) -> str:
        """
        Execute Phase 2 training (ranking post-training).
        
        Args:
            training_examples: List of training examples
            phase1_model_path: Path to Phase 1 trained model
            output_dir: Output directory for the trained model
            
        Returns:
            Path to the trained model
        """
        if not TRANSFORMERS_AVAILABLE or self.model is None:
            print("Training not available. Skipping Phase 2.")
            return self.model_name
        
        output_dir = output_dir or os.path.join(self.output_dir, 'phase2')
        os.makedirs(output_dir, exist_ok=True)
        
        # Load Phase 1 model if provided
        if phase1_model_path:
            print(f"Loading Phase 1 model from {phase1_model_path}")
            self.model = AutoModelForCausalLM.from_pretrained(phase1_model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(phase1_model_path)
        
        print(f"Starting Phase 2 training with {len(training_examples)} examples")
        
        # Convert to HuggingFace Dataset
        dataset = Dataset.from_list([
            {'text': ex.prompt + ex.completion} for ex in training_examples
        ])
        
        # Tokenize
        def tokenize_function(examples):
            return self.tokenizer(examples['text'], truncation=True, max_length=512)
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            warmup_steps=50,
            logging_steps=10,
            save_strategy='epoch',
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )
        
        # Train
        trainer.train()
        
        # Save model
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"Phase 2 training complete. Model saved to {output_dir}")
        return output_dir
    
    def create_sft_data(self, examples: List[TrainingExample]) -> List[Dict[str, str]]:
        """Create supervised fine-tuning data format"""
        return [
            {
                'instruction': ex.prompt,
                'output': ex.completion,
            }
            for ex in examples
        ]
    
    def create_preference_data(self, examples: List[TrainingExample]) -> List[Dict[str, str]]:
        """Create preference data for RLHF/DPO training"""
        preference_data = []
        for ex in examples:
            preference_data.append({
                'prompt': ex.prompt,
                'chosen': ex.completion,
                'rejected': '[]',  # Would need actual rejected completions
            })
        return preference_data
