"""LLM service for model management."""
from langchain_mistralai import ChatMistralAI
from langchain_openai import ChatOpenAI

from .config import ModelConfig, MISTRAL_API_KEY, OPENAI_API_KEY


class LLMService:
    """Service for managing LLM instances."""
    
    @staticmethod
    def get_mistral_llm() -> ChatMistralAI:
        """Get Mistral LLM instance.
        
        Returns:
            Configured Mistral LLM.
        """
        if not MISTRAL_API_KEY:
            raise ValueError("MISTRAL_API_KEY not found in environment variables")
        
        return ChatMistralAI(
            model=ModelConfig.MISTRAL_MODEL,
            api_key=MISTRAL_API_KEY,
            max_tokens=ModelConfig.MAX_TOKENS,
            temperature=ModelConfig.TEMPERATURE,
            timeout=120
        )
    
    @staticmethod
    def get_openai_llm() -> ChatOpenAI:
        """Get OpenAI LLM instance.
        
        Returns:
            Configured OpenAI LLM."""
        if not OPENAI_API_KEY:
            raise ValueError("OPEN_API_KEY not found in environment variables")
        
        return ChatOpenAI(
            model=ModelConfig.OPENAI_MODEL,
            api_key=OPENAI_API_KEY,
            max_tokens=ModelConfig.MAX_TOKENS,
            temperature=ModelConfig.TEMPERATURE,
            timeout=120
        )
    
    @staticmethod
    def get_llm(model_choice: str = "Mistral") -> ChatMistralAI:
        """Get LLM based on user choice.
        
        Args:
            model_choice: Name of the model to use.
            
        Returns:
            Configured LLM instance.
        """
        
        model_map = {
            "Mistral": LLMService.get_mistral_llm,
            "OpenAI": LLMService.get_openai_llm
        }
        
        if model_choice not in model_map:
            raise ValueError(
                f"Model '{model_choice}' not supported. "
                f"Available models: {', '.join(model_map.keys())}"
            )
        
        return model_map[model_choice]()