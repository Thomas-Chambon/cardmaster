"""Agent service for creating and managing LangChain agents."""
import streamlit as st
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_core.globals import set_llm_cache
from langgraph.checkpoint.memory import InMemorySaver
from langchain_community.cache import SQLiteCache

from .config import get_logger
from .llm_service import LLMService
from .rag_engine import RAGEngine, RAGConfig

logger = get_logger(__name__)

class AgentService:
    """Service for creating and managing agents."""
    
    def __init__(self):
        """Initialize the AgentService with RAG engine and LLM service."""
        self.rag_engine = RAGEngine()
        self.llm_service = LLMService()
    
    def _get_system_prompt(self, game_name: str) -> str:
        """Generate system prompt for the agent.
        
        Args:
            game_name: Name of the game.
            
        Returns:
            System prompt string.
        """

        prompt = (
            f"You are the 'CardMaster AI', a high-level judge and meta-game expert for {game_name}. "
            "Your goal is to provide surgical precision regarding rules, decklists, and market values.\n\n"

            "### PHASE 1: SEARCH PROTOCOL (MANDATORY)\n"
            f"1. For ANY question about {game_name}, you MUST start by calling `retrieve_context`.\n"
            "2. If the user asks for a price, value, or cost, You MUST use the `check_price` tool.\n"
            "3. Do not rely on your internal knowledge for facts, dates, or prices. Use the tools first.\n\n"

            "### PHASE 2: DATA PROCESSING & ANALYSIS\n"
            "- **Priority:** The retrieved context is your 'Single Source of Truth'.\n"
            "- **Recency:** You are in 2025. If context shows multiple versions of a rule or deck, use only the 2025 data.\n"
            "**- META & SYNERGIES:** When asked about deckbuilding or card strength, analyze synergies based on current 2025 competitive tiers. "
            "Explain WHY cards work together (e.g., mana curve, combo pieces, or board control).\n"
            "- **Integrity:** If the tool results are empty or irrelevant, say exactly: \"I don't have enough information in the current database to answer this question.\"\n"
            f"- **Pricing:** You MUST use the `check_price` tool to provide accurate pricing information. If an error occurs, ask the user to verify the card's spelling and precise the name must be in English.\n\n"

            "### PHASE 3: RESPONSE FORMULATING\n"
            "- **Language:** Detect the user's language (French or English) and respond in the same language.\n"
            "- **Style:** Analytical, professional, and structured. Use Markdown (bolding, lists) for readability.\n"
            "- **Citations:** Every factual statement must be followed by its source. End your response with a 'Sources' section.\n\n"

            "### OUTPUT STRUCTURE:\n"
            "1. **Direct Answer** (Concise and clear)\n"
            "2. **Detailed Analysis & Synergies** (Explain the interactions and meta-relevance)\n"
            "3. **Sources** (Format: [Source: Name of document/URL])\n\n"
            
            "Begin your analysis now."
        )

        return prompt
    
    def _create_price_tool(self, game_name: str) -> tool:
        """Create the price checking tool for the agent.

        Args:
            game_name: Name of the game.

        Returns:
            The price checking tool."""
        rag_engine = self.rag_engine

        @tool
        def check_price(card_name: str) -> str:
            """Check the market price of a card.

            Args:
                card_name: Name of the card to check.

            Returns:
                Market price information as a string.
            """

            return rag_engine.search_price_card(game_name, card_name)
        
        return check_price
        
    def _create_retrieval_tool(self) -> tool:
        """Create the retrieval tool for the agent.

        Returns:
            The retrieval tool.
        """
        rag_engine = self.rag_engine
        
        @tool
        def retrieve_context(query: str) -> str:
            """Retrieve information from game rules to help answer a query.
            
            Args:
                query: The user's question about game rules and deck building.
                
            Returns:
                Relevant context from the rules documentation.
            """
            return rag_engine.retrieve_context(query)
        
        return retrieve_context
    
    @st.cache_resource(ttl=3600, show_spinner=False)
    def create_agent(_self, game_name: str, model_choice: str = "Mistral") -> create_agent:
        """Create a configured agent for the specified game.
        
        Args:
            game_name: Name of the game (Magic The Gathering or Hearthstone).
            model_choice: LLM model to use.
            
        Returns:
            Configured agent.
        """
        # Initialize vector store (cached)
        _self.rag_engine.initialize_vector_store()
        
        # Get LLM
        llm = _self.llm_service.get_llm(model_choice)

        # Set LLM cache
        set_llm_cache(SQLiteCache(database_path=RAGConfig.LANGCHAIN_CACHE_PATH))
        
        # Create tools
        tools = [_self._create_retrieval_tool(), _self._create_price_tool(game_name)]
        
        # Get system prompt
        system_prompt = _self._get_system_prompt(game_name)

        
        # Create agent
        agent = create_agent(
            model=llm,
            tools=tools,
            system_prompt=system_prompt,
            checkpointer=InMemorySaver(),
        )
        
        return agent