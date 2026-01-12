"""
Dataset Classes for Knowledge-Point-Based Curriculum Learning

This module provides datasets that:
1. Load knowledge points and their difficulty levels
2. Generate prompts for the challenger with specific knowledge points
3. Support the standard RLHF training loop
"""

import json
import os
from typing import Any, Dict, List, Optional
from collections import defaultdict

import torch
import numpy as np
from torch.utils.data import Dataset
from datasets import load_dataset, Dataset as HFDataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from .knowledge_manager import KnowledgePointManager
from .prompts import build_knowledge_challenger_messages


def collate_fn(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function for dataloader."""
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)
    
    for feature in features:
        for key, value in feature.items():
            if isinstance(value, torch.Tensor):
                tensors[key].append(value)
            else:
                non_tensors[key].append(value)
    
    for key, value in tensors.items():
        tensors[key] = torch.stack(value, dim=0)
    
    for key, value in non_tensors.items():
        non_tensors[key] = np.array(value, dtype=object)
    
    return {**tensors, **non_tensors}


class KnowledgeChallengerDataset(Dataset):
    """
    Dataset for training the Challenger with knowledge-point-based prompts.
    
    Each item in the dataset represents a (knowledge_point, difficulty) pair
    that the challenger should use to generate a question.
    """
    
    def __init__(
        self,
        knowledge_manager: KnowledgePointManager,
        tokenizer: PreTrainedTokenizer,
        max_prompt_length: int = 2048,
        truncation: str = "right",
    ):
        """
        Initialize the dataset.
        
        Args:
            knowledge_manager: KnowledgePointManager instance
            tokenizer: Tokenizer for encoding prompts
            max_prompt_length: Maximum length of prompt tokens
            truncation: How to handle long prompts
        """
        self.knowledge_manager = knowledge_manager
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        
        # Get training batch (knowledge_point, difficulty pairs)
        self.training_batch = knowledge_manager.get_training_batch()
        
        print(f"KnowledgeChallengerDataset: {len(self.training_batch)} training samples")
    
    def refresh_batch(self):
        """Refresh the training batch (call after difficulty updates)."""
        self.training_batch = self.knowledge_manager.get_training_batch()
    
    def __len__(self):
        return len(self.training_batch)
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        kp, difficulty = self.training_batch[index]
        
        # Build messages
        messages = build_knowledge_challenger_messages(kp, difficulty)
        
        # Encode prompt
        if self.tokenizer.chat_template:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            )
        else:
            prompt = "system: " + messages[0]["content"] + '\n' + "user: " + messages[1]["content"]
        
        model_inputs = self.tokenizer(
            [prompt],
            add_special_tokens=False,
            return_tensors="pt"
        )
        input_ids = model_inputs["input_ids"][0]
        attention_mask = model_inputs["attention_mask"][0]
        
        # Handle truncation
        if len(input_ids) > self.max_prompt_length:
            if self.truncation == "left":
                input_ids = input_ids[-self.max_prompt_length:]
                attention_mask = attention_mask[-self.max_prompt_length:]
            elif self.truncation == "right":
                input_ids = input_ids[:self.max_prompt_length]
                attention_mask = attention_mask[:self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(input_ids)} > {self.max_prompt_length}")
        
        # Pad if necessary
        if len(input_ids) < self.max_prompt_length:
            pad_length = self.max_prompt_length - len(input_ids)
            # Left padding
            input_ids = torch.cat([
                torch.full((pad_length,), self.tokenizer.pad_token_id, dtype=input_ids.dtype),
                input_ids
            ])
            attention_mask = torch.cat([
                torch.zeros(pad_length, dtype=attention_mask.dtype),
                attention_mask
            ])
        
        # Position ids
        position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)
        
        # Raw prompt ids for logging
        raw_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length:]
            else:
                raw_prompt_ids = raw_prompt_ids[:self.max_prompt_length]
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "raw_prompt_ids": raw_prompt_ids,
            "ground_truth": "",  # No ground truth for challenger
            "knowledge_point": kp,
            "difficulty": difficulty,
        }


class KnowledgeSolverDataset(Dataset):
    """
    Dataset for training the Solver on generated questions.
    
    This dataset loads questions generated by the challenger, filtered by
    the uncertainty band (between beta and alpha solver accuracy).
    """
    
    def __init__(
        self,
        questions_path: str,
        tokenizer: PreTrainedTokenizer,
        max_prompt_length: int = 2048,
        truncation: str = "right",
        min_score: float = 0.3,
        max_score: float = 0.7,
    ):
        """
        Initialize the solver dataset.
        
        Args:
            questions_path: Path to JSON file with generated questions
            tokenizer: Tokenizer for encoding prompts
            max_prompt_length: Maximum prompt length
            truncation: Truncation strategy
            min_score: Minimum solver score (beta)
            max_score: Maximum solver score (alpha)
        """
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        
        # Load and filter questions
        self.data = []
        with open(questions_path, 'r', encoding='utf-8') as f:
            all_data = json.load(f)
        
        for item in all_data:
            score = item.get('score', 0)
            if min_score <= score <= max_score and item.get('answer'):
                self.data.append({
                    'problem': item['question'],
                    'answer': item['answer'],
                    'knowledge_point': item.get('knowledge_point', ''),
                    'difficulty': item.get('difficulty', 1),
                    'score': score,
                })
        
        print(f"KnowledgeSolverDataset: {len(self.data)} questions (filtered from {len(all_data)})")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = self.data[index]
        question = item['problem']
        answer = item['answer']
        
        # Build solver prompt
        messages = [
            {"role": "system", "content": r"Please reason step by step, and put your final answer within \boxed{}."},
            {"role": "user", "content": question}
        ]
        
        # Encode
        if self.tokenizer.chat_template:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            )
        else:
            prompt = "system: " + messages[0]["content"] + '\n' + "user: " + messages[1]["content"]
        
        model_inputs = self.tokenizer(
            [prompt],
            add_special_tokens=False,
            return_tensors="pt"
        )
        input_ids = model_inputs["input_ids"][0]
        attention_mask = model_inputs["attention_mask"][0]
        
        # Handle truncation
        if len(input_ids) > self.max_prompt_length:
            if self.truncation == "left":
                input_ids = input_ids[-self.max_prompt_length:]
                attention_mask = attention_mask[-self.max_prompt_length:]
            elif self.truncation == "right":
                input_ids = input_ids[:self.max_prompt_length]
                attention_mask = attention_mask[:self.max_prompt_length]
        
        # Pad if necessary
        if len(input_ids) < self.max_prompt_length:
            pad_length = self.max_prompt_length - len(input_ids)
            input_ids = torch.cat([
                torch.full((pad_length,), self.tokenizer.pad_token_id, dtype=input_ids.dtype),
                input_ids
            ])
            attention_mask = torch.cat([
                torch.zeros(pad_length, dtype=attention_mask.dtype),
                attention_mask
            ])
        
        position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)
        
        raw_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            raw_prompt_ids = raw_prompt_ids[:self.max_prompt_length]
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "raw_prompt_ids": raw_prompt_ids,
            "ground_truth": answer,
            "knowledge_point": item.get('knowledge_point', ''),
            "difficulty": item.get('difficulty', 1),
        }


def create_knowledge_challenger_dataloader(
    knowledge_manager: KnowledgePointManager,
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 512,
    max_prompt_length: int = 2048,
    shuffle: bool = True,
    seed: int = 42,
) -> torch.utils.data.DataLoader:
    """Create dataloader for challenger training."""
    from torch.utils.data import RandomSampler, SequentialSampler
    from torchdata.stateful_dataloader import StatefulDataLoader
    
    dataset = KnowledgeChallengerDataset(
        knowledge_manager=knowledge_manager,
        tokenizer=tokenizer,
        max_prompt_length=max_prompt_length,
    )
    
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(seed)
        sampler = RandomSampler(dataset, generator=generator)
    else:
        sampler = SequentialSampler(dataset)
    
    dataloader = StatefulDataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=True,
    )
    
    return dataloader


def create_knowledge_solver_dataloader(
    questions_path: str,
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 512,
    max_prompt_length: int = 2048,
    min_score: float = 0.3,
    max_score: float = 0.7,
    shuffle: bool = True,
    seed: int = 42,
) -> torch.utils.data.DataLoader:
    """Create dataloader for solver training."""
    from torch.utils.data import RandomSampler, SequentialSampler
    from torchdata.stateful_dataloader import StatefulDataLoader
    
    dataset = KnowledgeSolverDataset(
        questions_path=questions_path,
        tokenizer=tokenizer,
        max_prompt_length=max_prompt_length,
        min_score=min_score,
        max_score=max_score,
    )
    
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(seed)
        sampler = RandomSampler(dataset, generator=generator)
    else:
        sampler = SequentialSampler(dataset)
    
    dataloader = StatefulDataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=True,
    )
    
    return dataloader
