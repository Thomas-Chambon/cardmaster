"""RAG engine for document retrieval."""
from typing import List
import streamlit as st
from chromadb import PersistentClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tavily import TavilyClient
from urllib.parse import quote
import requests, hashlib, json, os
from dotenv import load_dotenv, set_key, find_dotenv

env_path = find_dotenv()
load_dotenv()

from .config import RAGConfig, TAVILY_API_KEY, HASH_CONFIG_FILE, CONFIG_FILE, get_logger
from .document_loader import DocumentLoader

logger = get_logger(__name__)

class RAGEngine:
    """Retrieval-Augmented Generation engine."""
    
    def __init__(self):
        self.embeddings = None
        self.vector_store = None
        self.document_loader = DocumentLoader()
        logger.info("RAGEngine initialized")
    
    def _initialize_embeddings(self):
        """Initialize embedding model."""
        
        if self.embeddings is None:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=RAGConfig.EMBEDDING_MODEL
            )

    def compare_configs(self, original_hash = HASH_CONFIG_FILE, config_file = CONFIG_FILE) -> bool:
        """
        Compare the new configuration with the existing one to check for changes.

        Args:
            new_config: The new configuration dictionary.

        Returns:
            True if configurations differ, False otherwise.
        """

        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                new_config = json.load(f)
                hash_new_config = hashlib.sha256(json.dumps(new_config, sort_keys=True).encode('utf-8')).hexdigest()

        except FileNotFoundError:
            logger.info("No existing config hash file found. Assuming no changes.")
            return True

        if original_hash != hash_new_config:
            logger.info("Configuration changes detected.")
            return True
        
        logger.info("No configuration changes detected.")
        return False

    def _split_documents(self, docs: List[Document]) -> List[Document]:
        """
        Split documents into chunks using a fast recursive approach.
        This version removes SemanticChunker for maximum performance.

        Args:
            docs: List of documents to split.

        Returns:
            List of document chunks.
        """
        logger.info("Starting document splitting process...")
        
        # Standard fast recursive splitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=RAGConfig.CHUNK_SIZE,
            chunk_overlap=RAGConfig.CHUNK_OVERLAP,
            separators=RAGConfig.SEPARATORS,
        )

        with st.status("Ingesting knowledge base...", expanded=True) as status:
            st.write("Initializing text splitter...")
            
            final_chunks = []
            total_docs = len(docs)
            
            progress_bar = st.progress(0, text="Processing documents...")
            
            for i, doc in enumerate(docs):
                # Split the individual document
                doc_chunks = splitter.split_documents([doc])
                final_chunks.extend(doc_chunks)
                
                # Update progress bar
                progress = (i + 1) / total_docs
                progress_bar.progress(
                    progress, 
                    text=f"Splitting: {int(progress * 100)}% (Document {i+1}/{total_docs})"
                )
                
            # Final UI updates
            progress_bar.empty()
            status.update(
                label=f"✅ Success! {len(final_chunks)} chunks generated.", 
                state="complete", 
                expanded=False
            )

        logger.info(f"Splitting complete. Total chunks created: {len(final_chunks)}")
        return final_chunks
    
    def generate_ids(self, final_chunks: List[Document]) -> List[str]:
        """Generate unique IDs for each document chunk.
        Args:
            final_chunks: List of document chunks.
        Returns:
            List of unique IDs.
        """
        logger.info("Generating unique IDs for document chunks...")
        ids = []
        for i, chunk in enumerate(final_chunks):
            unique_string = f"{i}-{chunk.page_content}"
            chunk_id = hashlib.sha256(unique_string.encode('utf-8')).hexdigest()
            ids.append(chunk_id)

        logger.info("ID generation complete for all chunks.")
        return ids

    @st.cache_resource(ttl=None)
    def initialize_vector_store(_self) -> Chroma:
        """Initialize or load the Chroma vector store.

        Returns:
            Chroma vector store instance.
        """
        if _self.vector_store is not None:
            return _self.vector_store
        
        embeddings = _self._initialize_embeddings()
        client = PersistentClient(path=RAGConfig.CHROMA_PATH)
        
        # Load current config for hashing
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            current_config_data = json.load(f)
        
        # New hash calculation
        new_hash = hashlib.sha256(json.dumps(current_config_data, sort_keys=True).encode('utf-8')).hexdigest()
        
        # Check hash to see if sources changed
        stored_hash = os.getenv("HASH_CONFIG_FILE")
        update_sources = (stored_hash != new_hash)

        # Chroma vector store initialization
        vector_store = Chroma(
            client=client,
            collection_name="cardmaster_index",
            embedding_function=embeddings
        )

        # Update or initialize vector store if needed
        if update_sources or vector_store._collection.count() == 0:
            reason = "Sources changed, updating" if update_sources else "Index empty, initializing"
            logger.info(f"{reason} vector store...")
            st.info(f"{reason}, this may take some time.")

            documents = _self.document_loader.load_all_documents()
            chunks = _self._split_documents(documents)
            ids = _self.generate_ids(chunks)

            # Add documents to vector store
            vector_store.add_documents(documents=chunks, ids=ids)
            
            # Update stored hash
            set_key(find_dotenv(), "HASH_CONFIG_FILE", new_hash)
            st.success("Index updated and hash saved.")

        else:
            logger.info("Loading existing vector store from disk.")
            st.success("Index loaded from disk (no changes detected).")

        _self.vector_store = vector_store
        return _self.vector_store
    
    def retrieve_context(self, query: str, k: int = None) -> str:
        """Retrieve relevant context for a query.
        
        Args:
            query: User query.
            k: Number of documents to retrieve.
            
        Returns:
            Serialized context from retrieved documents.
        """
        if self.vector_store is None:
            self.vector_store = self.initialize_vector_store()
        
        k = k or RAGConfig.SIMILARITY_TOP_K
        retrieved_docs = self.vector_store.similarity_search(query, k=k)
        
        # Serialize results
        serialized = "\n\n".join(
            f"Source: {doc.metadata.get('source', 'Unknown')}\n"
            f"Content: {doc.page_content}"
            for doc in retrieved_docs
        )
        
        return serialized

    def search_price_card(self, game_name: str, card_name: str) -> str:
        """
        Get market prices (MTG) or craft costs/meta data (Hearthstone).
        Args:
            game_name: The name of the game.
            card_name: The exact name of the card.
        Returns:
            Formatted string with price or craft cost information.
        """

        # MAGIC THE GATHERING (Scryfall API)
        if game_name == "Magic The Gathering":
            logger.info(f"Searching MTG price for card: {card_name}")
            try:
                url = f"https://api.scryfall.com/cards/named?exact={quote(card_name)}"
                response = requests.get(url, timeout=5)

                if response.status_code == 200:
                    data = response.json()
                    p = data.get('prices', {})
                    return (f"MTG Price for {data['name']}:\n"
                            f"- Normal: {p.get('eur', 'N/A')}€ | Foil: {p.get('eur_foil', 'N/A')}€\n"
                            f"- View: {data.get('scryfall_uri')}")
                logger.info(f"Card '{card_name}' not found on Scryfall.")

            except Exception as e:
                logger.error(f"MTG API Error: {str(e)}")

        # HEARTHSTONE (Tavily Search)
        elif game_name == "Hearthstone":
            logger.info(f"Searching Hearthstone data for card: {card_name}")
            try:
                
                tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
                raw_results = tavily_client.search(f"Find the \"cost to craft\" for {card_name} Hearthstone card", max_results=3)
                    
                logger.info(f"Tavily raw results: {raw_results}")

                # Format results
                if len(raw_results['results']) > 0:
                    details = "\n".join([f"- {res['content']} (Source: {res['url']})" for res in raw_results['results']])
                        
                else:
                    details = "No cost to craft information found."

                link = f"https://www.hearthpwn.com/cards?filter-name={card_name}" 
                return (f"Hearthstone Info for '{card_name}':\n{details}\n"
                        f"- Source Link: {link}")

            except Exception as e:
                logger.error(f"Tavily Error: {str(e)}")
                return f"Manual search link for {card_name}: https://www.hearthpwn.com/cards?filter-name={card_name}"

            
    
