"""
Utility functions for anon-repo data processing.
"""

import re
import json
import hashlib
from typing import List, Dict, Any, Optional
from pathlib import Path

def clean_text(text: str) -> str:
    """Clean and normalize text content."""
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters that might cause issues
    text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\"\']', '', text)
    
    return text.strip()

def extract_prompt_from_conversation(text: str) -> str:
    """Extract the human prompt from a conversation format."""
    # Common patterns for extracting prompts
    patterns = [
        r"Human:\s*(.*?)(?:\n\nAssistant:|$)",
        r"User:\s*(.*?)(?:\n\nAssistant:|$)",
        r"Question:\s*(.*?)(?:\n\nAnswer:|$)",
        r"Prompt:\s*(.*?)(?:\n\nResponse:|$)"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return clean_text(match.group(1))
    
    # If no pattern matches, return the original text
    return clean_text(text)

def generate_prompt_id(prompt: str, source: str) -> str:
    """Generate a unique ID for a prompt."""
    content = f"{source}:{prompt}"
    return hashlib.md5(content.encode()).hexdigest()

def validate_prompt_data(prompt_data: Dict[str, Any]) -> bool:
    """Validate that prompt data contains required fields."""
    required_fields = ['prompt', 'label', 'source']
    
    for field in required_fields:
        if field not in prompt_data or not prompt_data[field]:
            return False
    
    # Validate label
    if prompt_data['label'] not in ['harmful', 'benign']:
        return False
    
    # Validate prompt length
    if len(prompt_data['prompt'].strip()) < 10:
        return False
    
    return True

def deduplicate_prompts(prompts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicate prompts based on content similarity."""
    seen_prompts = set()
    unique_prompts = []
    
    for prompt_data in prompts:
        prompt_text = prompt_data['prompt'].lower().strip()
        
        if prompt_text not in seen_prompts:
            seen_prompts.add(prompt_text)
            unique_prompts.append(prompt_data)
    
    return unique_prompts

def split_dataset(prompts: List[Dict[str, Any]], train_ratio: float = 0.8, 
                 val_ratio: float = 0.1, test_ratio: float = 0.1) -> Dict[str, List[Dict[str, Any]]]:
    """Split dataset into train, validation, and test sets."""
    import random
    
    # Shuffle the data
    random.shuffle(prompts)
    
    total_size = len(prompts)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    
    train_data = prompts[:train_size]
    val_data = prompts[train_size:train_size + val_size]
    test_data = prompts[train_size + val_size:]
    
    return {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }

def save_jsonl(data: List[Dict[str, Any]], filepath: str):
    """Save data to JSONL format."""
    import jsonlines
    
    with jsonlines.open(filepath, 'w') as writer:
        for item in data:
            writer.write(item)

def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """Load data from JSONL format."""
    import jsonlines
    
    data = []
    with jsonlines.open(filepath) as reader:
        for item in reader:
            data.append(item)
    
    return data

def create_data_summary(prompts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create a summary of the dataset."""
    harmful_count = sum(1 for p in prompts if p['label'] == 'harmful')
    benign_count = sum(1 for p in prompts if p['label'] == 'benign')
    
    # Calculate average prompt length
    avg_length = sum(len(p['prompt']) for p in prompts) / len(prompts) if prompts else 0
    
    # Get unique sources
    sources = list(set(p['source'] for p in prompts))
    
    return {
        'total_prompts': len(prompts),
        'harmful_prompts': harmful_count,
        'benign_prompts': benign_count,
        'harmful_ratio': harmful_count / len(prompts) if prompts else 0,
        'benign_ratio': benign_count / len(prompts) if prompts else 0,
        'average_prompt_length': avg_length,
        'sources': sources,
        'unique_sources_count': len(sources)
    }
