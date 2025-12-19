"""Reusable UI components for Streamlit."""
import streamlit as st

def render_sidebar():
    """Render the sidebar with app information."""
    st.sidebar.title("CardMaster AI")
    st.sidebar.markdown("---")
    
    st.sidebar.header("About")
    st.sidebar.markdown("""
    Your intelligent assistant for **Magic The Gathering** and **Hearthstone**.
    
    Whether you're a beginner or an experienced player, CardMaster AI supports you 
    in your gaming experience.
    """)
    
    st.sidebar.markdown("---")
    
    st.sidebar.header("Features")
    
    with st.sidebar.expander("Magic The Gathering"):
        st.markdown("""
        - Official rules reference
        - Card interaction analysis
        - Deck recommendations by format
        - Real-time prices via Cardmarket
        - Sideboard advice
        """)
    
    with st.sidebar.expander("Hearthstone"):
        st.markdown("""
        - Game rules and mechanics
        - Deck suggestions by class
        - Synergy analysis
        - Prices via Hearthpwn
        - Arena & Battlegrounds tips
        """)
    
    st.sidebar.markdown("---")
    
    st.sidebar.header("Examples of use")
    st.sidebar.markdown("""
    **Questions possibles / Possible questions:**
    - "Comment fonctionne le flashback ?"
    - "Quel deck Druide pour la méta actuelle ?"
    - "Prix d'un Black Lotus ?"
    - "Meilleure synergie pour un deck Démon ?"
    - “How does flashback work?”
    - “Which Druid deck for the current meta?”
    - “How much does a Black Lotus cost?”
    - “Best synergy for a Demon deck?”
    """)
    
    st.sidebar.markdown("---")
    
    st.sidebar.header("Technical Stack")
    st.sidebar.markdown("""
    * **Framework:** [LangChain](https://www.langchain.com/)
    * **LLMs:** Mistral AI (`mistral-large-latest`) & OpenAI (`gpt-4.1`) (HuggingFace)
    * **Vector Database:** [ChromaDB](https://www.trychroma.com/) (Persistent storage)
    * **Embeddings:** `Qwen/Qwen3-Embedding-8B` (HuggingFace)
    * **UI:** [Streamlit](https://streamlit.io/)
    """)
    
    st.sidebar.markdown("---")
    
    st.sidebar.info("**Tip: Ask specific questions to get the best answers!**")
    
    # Clear history button
    if st.sidebar.button("Clear Chat history", use_container_width=True):
        st.session_state.messages = []
        st.success('Chat history cleared!', icon="✅")

    # Clear cache chat button
    if st.sidebar.button('Refresh Cache', use_container_width=True):
        st.cache_resource.clear()
        st.success('Cache clear!', icon="✅")
    
    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Developed by [Thomas Chambon](https://github.com/Thomas-Chambon) | "
        "2025 CardMaster AI | v1.0"
    )


def render_game_selector():
    """Render game selection radio buttons.
    
    Returns:
        Selected game name.
    """
    game = st.radio(
        "Please select your favorite game :", 
        ("Magic The Gathering", "Hearthstone"),
        key="game_radio"
    )
    return game


def render_model_selector():
    """Render model selection dropdown.
    
    Returns:
        Selected model name.
    """
    llm_choice = st.selectbox(
        "Please select the language model",
        ("Mistral", "OpenAI"),
        key="llm_user_select"
    )
    return llm_choice


def render_chat_message(role: str, content: str):
    """Render a chat message.
    
    Args:
        role: Message role (user or assistant).
        content: Message content.
    """
    st.chat_message(role).markdown(content)


def render_download_button(content: str, filename: str):
    """Render download button for response.
    
    Args:
        content: Content to download.
        filename: Name of the download file.
    """
    st.download_button(
        label="Exporter en JSON",
        data=content,
        file_name=f"{filename}.json",
        mime="text/json",
        icon=":material/download:",
        key=f"download_{filename}"
    )