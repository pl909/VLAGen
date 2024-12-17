"""Simplified Phi implementation to avoid dependency issues"""
from .base_llm import LLMBackbone

class PhiLLMBackbone(LLMBackbone):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Phi support is disabled to avoid dependency issues")
