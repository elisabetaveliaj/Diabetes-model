# baseline.py


import os
from pathlib import Path
from typing import Optional, List, Dict, Any
from openai import OpenAI
import yaml

BASE_DIR = Path(__file__).resolve().parent
_CONFIG_PATH = BASE_DIR / "data" / "config.yaml"

try:
    with open(_CONFIG_PATH, "r") as f:
        _CONFIG = yaml.safe_load(f) or {}
except FileNotFoundError:
    _CONFIG = {}


_OPENAI_CLIENT = None


def _get_openai_client():
    """Get or create OpenAI client instance."""
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            # Try to get from config
            api_key = _CONFIG.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables or config")
        _OPENAI_CLIENT = OpenAI(api_key=api_key)
    return _OPENAI_CLIENT





def run_baseline(prompt: str, model: str = "gpt-3.5-turbo", temperature: float = 0.5) -> Dict[str, Any]:
    """
    Query OpenAI model as a baseline - just the prompt, no system message or context.

    """
    client = _get_openai_client()
    
    messages = [{"role": "user", "content": prompt}]
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature
        )
        
        return {
            "response": response.choices[0].message.content,
            "model": model,
            "retrieved_facts": [],  # Always empty for baseline
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        }
    except Exception as e:
        return {
            "response": f"Error calling OpenAI API: {str(e)}",
            "model": model,
            "retrieved_facts": [],
            "error": str(e)
        }
