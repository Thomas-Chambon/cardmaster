"""Configuration and constants for CardMaster AI."""
import os
import logging, json
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, List
from dotenv import load_dotenv

load_dotenv()

# Paths
BASE_DIR = Path(__file__).parent.parent
RAG_SOURCES_DIR = BASE_DIR / "rag_sources"
CONFIG_FILE = RAG_SOURCES_DIR / "config.json"

# API Keys
MISTRAL_API_KEY = os.getenv("MISTRALAI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
USER_AGENT = os.getenv("USER_AGENT", "CardMasterAI/1.0")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Model Configuration
class ModelConfig:
    """LLM model configuration."""

    MISTRAL_MODEL = "mistral-large-latest"
    OPENAI_MODEL = "gpt-4.1"
    MAX_TOKENS = 2000
    TEMPERATURE = 0.0

# RAG Configuration
class RAGConfig:
    """RAG system configuration."""

    EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
    CHUNK_SIZE = 1500 # Number of characters per chunk
    CHUNK_OVERLAP = 200 # Number of overlapping characters between chunks
    SEPARATORS = ["\n\n", "\n", ". ", " ", ""]
    SIMILARITY_TOP_K = 7
    CHROMA_PATH = "./chroma_db_cache"
    LANGCHAIN_CACHE_PATH = "./langchain.db"

# Game Configuration
class GameConfig:
    """Game-specific configuration."""
    
    GAMES = {
        "Magic The Gathering": {
            "price_site": "https://www.cardmarket.com",
            "display_name": "Magic The Gathering"
        },
        "Hearthstone": {
            "price_site": "https://www.hearthpwn.com",
            "display_name": "Hearthstone"
        }
    }

def load_sources_config() -> Dict[str, List[str]]:
    """Load URLs and PDF paths from config file.
    
    Returns:
        Dictionary with 'urls' and 'pdfs' lists.
    """
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(f"Config file not found: {CONFIG_FILE}")
    
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    return config

def get_logger(name: str = "cardmasterAI") -> logging.Logger:
    """Set up and return a logger.
    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    log_file = (BASE_DIR / 'cardmaster_ai.log')
    file_handler = RotatingFileHandler(log_file, mode='a', maxBytes=5*1024*1024, 
                                 backupCount=1, encoding=None, delay=False)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger