"""Dummy LLM backbone implementation"""
from typing import Any

class LLMBackbone:
    """Stub class for LLM backbone"""
    def __init__(self, *args: Any, **kwargs: Any):
        pass

    def get_tokenizer(self):
        """Return dummy tokenizer"""
        return None

# Create dummy classes for each LLM type
class PhiLLMBackbone(LLMBackbone):
    pass

class LLaMa2LLMBackbone(LLMBackbone):
    pass

class MistralLLMBackbone(LLMBackbone):
    pass

__all__ = ["LLMBackbone", "PhiLLMBackbone", "LLaMa2LLMBackbone", "MistralLLMBackbone"]
