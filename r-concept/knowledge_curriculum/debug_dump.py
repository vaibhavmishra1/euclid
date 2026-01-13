"""
Debug Dump Utility for Training Data

This module provides functionality to dump:
1. All prompts sent to challenger
2. All challenger outputs
3. All solver responses

Controlled by environment variable: DUMP_DEBUG_DATA (set to "1" or "true" to enable)
"""

import os
import json
import threading
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path


class DebugDumper:
    """Thread-safe debug data dumper."""
    
    def __init__(self, base_path: Optional[str] = None):
        """
        Initialize debug dumper.
        
        Args:
            base_path: Base directory for dump files. Defaults to STORAGE_PATH/debug_dumps
        """
        self.enabled = os.getenv("DUMP_DEBUG_DATA", "0").lower() in ("1", "true", "yes")
        
        # Initialize attributes that might be needed even if disabled
        self.lock = threading.Lock()
        self.current_iteration = None
        self.current_step = None
        self.max_string_length = 10000  # 10k chars per field
        
        if not self.enabled:
            self.base_path = None
            self.prompts_file = None
            self.challenger_outputs_file = None
            self.solver_responses_file = None
            return
        
        if base_path is None:
            storage_path = os.getenv("STORAGE_PATH", "/workspace/rzero_storage")
            base_path = os.path.join(storage_path, "debug_dumps")
        
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # File handles (opened on first write)
        self.prompts_file = None
        self.challenger_outputs_file = None
        self.solver_responses_file = None
        
        print(f"DEBUG DUMP: Enabled. Dumping to {self.base_path}")
    
    def is_enabled(self) -> bool:
        """Check if debug dumping is enabled."""
        return self.enabled
    
    def _truncate_string(self, s: str, max_length: Optional[int] = None) -> str:
        """Truncate a string if it exceeds max_length."""
        if max_length is None:
            max_length = self.max_string_length
        if len(s) > max_length:
            return s[:max_length] + f"... [TRUNCATED: {len(s)} chars total]"
        return s
    
    def _truncate_dict_values(self, d: Dict[str, Any], max_length: Optional[int] = None) -> Dict[str, Any]:
        """Recursively truncate string values in a dictionary."""
        if max_length is None:
            max_length = self.max_string_length
        result = {}
        for key, value in d.items():
            if isinstance(value, str):
                result[key] = self._truncate_string(value, max_length)
            elif isinstance(value, dict):
                result[key] = self._truncate_dict_values(value, max_length)
            elif isinstance(value, list):
                result[key] = [
                    self._truncate_string(item, max_length) if isinstance(item, str)
                    else self._truncate_dict_values(item, max_length) if isinstance(item, dict)
                    else item
                    for item in value[:10]  # Limit list to first 10 items
                ] + (["... [TRUNCATED: more items]"] if len(value) > 10 else [])
            else:
                result[key] = value
        return result
    
    def set_iteration(self, iteration: int):
        """Set current iteration number."""
        if not self.enabled:
            return
        
        with self.lock:
            if self.current_iteration != iteration:
                self._close_files()
                self.current_iteration = iteration
                self.current_step = None
                print(f"DEBUG DUMP: Starting iteration {iteration}")
    
    def set_step(self, step: int):
        """Set current training step number."""
        if not self.enabled:
            return
        
        with self.lock:
            if self.current_step != step:
                self._close_files()
                self.current_step = step
                print(f"DEBUG DUMP: Starting step {step}")
    
    def _close_files(self):
        """Close all open files."""
        for f in [self.prompts_file, self.challenger_outputs_file, self.solver_responses_file]:
            if f is not None:
                try:
                    f.close()
                except:
                    pass
        
        self.prompts_file = None
        self.challenger_outputs_file = None
        self.solver_responses_file = None
    
    def _get_file_path(self, file_type: str) -> Path:
        """Get file path for a given type."""
        if self.current_iteration is not None:
            if self.current_step is not None:
                filename = f"iter_{self.current_iteration}_step_{self.current_step}_{file_type}.jsonl"
            else:
                filename = f"iter_{self.current_iteration}_{file_type}.jsonl"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{file_type}.jsonl"
        
        return self.base_path / filename
    
    def dump_prompt(self, prompt: str, metadata: Dict[str, Any]):
        """
        Dump a prompt sent to challenger.
        
        Args:
            prompt: The formatted prompt text
            metadata: Additional metadata (knowledge_points, difficulty, set_id, etc.)
        """
        if not self.enabled:
            return
        
        with self.lock:
            if self.prompts_file is None:
                file_path = self._get_file_path("prompts")
                self.prompts_file = open(file_path, 'a', encoding='utf-8')
            
            # Truncate large strings
            truncated_prompt = self._truncate_string(prompt)
            truncated_metadata = self._truncate_dict_values(metadata)
            
            entry = {
                "timestamp": datetime.now().isoformat(),
                "prompt": truncated_prompt,
                **truncated_metadata
            }
            self.prompts_file.write(json.dumps(entry, ensure_ascii=False) + '\n')
            self.prompts_file.flush()
    
    def dump_challenger_output(self, output: str, metadata: Dict[str, Any]):
        """
        Dump challenger output.
        
        Args:
            output: Raw challenger output text
            metadata: Additional metadata (parsed question, answer, set_id, etc.)
        """
        if not self.enabled:
            return
        
        with self.lock:
            if self.challenger_outputs_file is None:
                file_path = self._get_file_path("challenger_outputs")
                self.challenger_outputs_file = open(file_path, 'a', encoding='utf-8')
            
            # Truncate large strings
            truncated_output = self._truncate_string(output)
            truncated_metadata = self._truncate_dict_values(metadata)
            
            entry = {
                "timestamp": datetime.now().isoformat(),
                "raw_output": truncated_output,
                **truncated_metadata
            }
            self.challenger_outputs_file.write(json.dumps(entry, ensure_ascii=False) + '\n')
            self.challenger_outputs_file.flush()
    
    def dump_solver_response(self, question: str, response: Dict[str, Any]):
        """
        Dump solver response for a question.
        
        Args:
            question: The question sent to solver
            response: Solver response dict (answer, score, results, etc.)
        """
        if not self.enabled:
            return
        
        with self.lock:
            if self.solver_responses_file is None:
                file_path = self._get_file_path("solver_responses")
                self.solver_responses_file = open(file_path, 'a', encoding='utf-8')
            
            # Truncate large strings
            truncated_question = self._truncate_string(question)
            truncated_response = self._truncate_dict_values(response)
            
            entry = {
                "timestamp": datetime.now().isoformat(),
                "question": truncated_question,
                **truncated_response
            }
            self.solver_responses_file.write(json.dumps(entry, ensure_ascii=False) + '\n')
            self.solver_responses_file.flush()
    
    def dump_batch(self, prompts: List[str], outputs: List[str], metadata_list: List[Dict[str, Any]]):
        """
        Dump a batch of prompts and outputs.
        
        Args:
            prompts: List of prompts
            outputs: List of challenger outputs
            metadata_list: List of metadata dicts (one per item)
        """
        if not self.enabled:
            return
        
        for prompt, output, metadata in zip(prompts, outputs, metadata_list):
            self.dump_prompt(prompt, metadata)
            self.dump_challenger_output(output, metadata)
    
    def close(self):
        """Close all files and cleanup."""
        if not self.enabled:
            return
        
        with self.lock:
            self._close_files()


# Global instance
_global_dumper: Optional[DebugDumper] = None


def get_dumper() -> DebugDumper:
    """Get or create global debug dumper instance."""
    global _global_dumper
    if _global_dumper is None:
        _global_dumper = DebugDumper()
    return _global_dumper


def is_enabled() -> bool:
    """Check if debug dumping is enabled."""
    return os.getenv("DUMP_DEBUG_DATA", "0").lower() in ("1", "true", "yes")
