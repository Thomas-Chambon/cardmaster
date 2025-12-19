"""Main Streamlit application."""
import os
import streamlit as st
from dotenv import load_dotenv

# Adjust path for module imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.agent_service import AgentService
from src.config import USER_AGENT, get_logger
from ui.components import (
    render_sidebar,
    render_game_selector,
    render_model_selector,
    render_chat_message,
    render_download_button
)

# Initialize logger
logger = get_logger(__name__)

# Load environment variables
load_dotenv()
os.environ["USER_AGENT"] = USER_AGENT

@st.cache_resource
def get_agent_service():
    """Initialise et met en cache l'objet AgentService."""
    return AgentService()

def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    if 'agent' not in st.session_state:
        st.session_state.agent = None
    
    if 'current_game' not in st.session_state:
        st.session_state.current_game = None

def get_or_create_agent(game_name: str, model_choice: str):
    """Get existing agent or create new one if game or model changed.
    
    Args:
        game_name: Selected game name.
        model_choice: Selected model choice.
    
    Returns:
        Configured agent.
    """
    # Initialize session state variables if needed
    if 'current_game' not in st.session_state:
        st.session_state.current_game = None
    if 'model_choice' not in st.session_state:
        st.session_state.model_choice = None
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    
    # Check if recreation is needed
    game_changed = st.session_state.current_game != game_name
    model_changed = st.session_state.model_choice != model_choice
    needs_recreation = game_changed or model_changed or st.session_state.agent is None
    
    # Skip if nothing changed
    if not needs_recreation:
        return st.session_state.agent
    
    # Determine spinner message
    if st.session_state.agent is None:
        message = "Agent initialization..."
    elif game_changed:
        message = "Agent initialization..."
    else:
        message = "Agent re-initialization due to model change..."
    
    # Create or recreate agent
    with st.spinner(message, show_time=True):
        st.session_state.agent = st.session_state.agent_service.create_agent(
            game_name=game_name,
            model_choice=model_choice
        )
        st.session_state.current_game = game_name
        st.session_state.model_choice = model_choice
    
    st.info(f"Using model: {model_choice} and game: {game_name}", icon="🤖")
    return st.session_state.agent

def main():
    """Main application function.
    
    Returns:
        None
    """
    # Page configuration
    st.set_page_config(page_title="CardMaster AI", page_icon="img/favicon.ico", layout="wide")

    # Initialize AgentService
    agent_service = get_agent_service()
    st.session_state.agent_service = agent_service

    # Initialize session state
    initialize_session_state()
    
    # Render sidebar
    render_sidebar()
    
    # Main content
    st.title("CardMaster AI : Your Intelligent Assistant for Magic The Gathering and Hearthstone") 
    st.caption("Ask questions in English or in French 🇫🇷 🇬🇧 (FR / EN)")
    # User selections
    col1, col2 = st.columns(2)
    with col1:
        game = render_game_selector()
    with col2:
        llm_choice = render_model_selector()
    
    st.markdown("---")

    # Get or create agent
    agent = get_or_create_agent(game, llm_choice)

    # Chat container
    chat_container = st.container()
    
    # Display chat history
    with chat_container:
        for message in st.session_state.messages:
            render_chat_message(message["role"], message["content"])
    
    # Chat input
    query = st.chat_input("Ask your question!", key="chat_input")
    
    if query:
        # Display user message
        render_chat_message("user", query)
        st.session_state.messages.append({"role": "user", "content": query})
        
        # Get agent response
        with st.spinner("Agent thinking...", show_time=True):
            try:
                result = agent.invoke(
                    {"messages": [{"role": "user", "content": query}]},
                    config = {"configurable": {"thread_id": "1"}}
                )
                response = result["messages"][-1].content
                
                # Display assistant message
                render_chat_message("assistant", response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                # Download button
                render_download_button(response, query[:30])
                
            except Exception as e:
                st.error(f"Erreur lors de la génération de la réponse : {e}")


if __name__ == "__main__":
    logger.info("Starting CardMaster AI...")
    main()