# CardMaster AI: RAG System for CCG Strategy

CardMaster AI is a high-performance Retrieval-Augmented Generation (RAG) assistant. It is specifically engineered to help Magic: The Gathering and Hearthstone players navigate complex rules, analyze deck synergies, and monitor market trends with low-latency responses.

## Key Features

* **Recursive Chunking:** It uses logical delimiters (paragraphs, sentences) to maintain rule integrity while ensuring near-instant document processing.

* **Real-Time UI Feedback:** Integrated Streamlit progress bars and status containers to monitor the ingestion and indexing process in real-time.

* **Decision-Making Agent:** A sophisticated agent that determines whether to query the local knowledge base for rules or use external tools for live data.

* **Market Intelligence:** Built-in tools for Scryfall API integration (MTG) and automated link generation for Hearthstone card databases.

## Project Structure

```text
CardMasterAI/
├── chroma_db_cache/      # Local persistent vector database (Auto-generated)
├── rag_sources/          # Source documents and configuration
│   ├── config.json       # JSON file listing URLs and PDF paths
│   └── rules_mtg.pdf     # Example PDF source
├── src/
│   ├── agent_service.py  # Agent orchestration and tool definitions
│   ├── config.py         # Global settings, logging, and game configurations
│   ├── document_loader.py # Data loading logic (PDF, Web scraping)
│   ├── llm_service.py    # LLM API connection manager
│   └── rag_engine.py     # Indexing engine and semantic search logic
├── ui/
│   ├── app.py            # Main Streamlit application entry point
│   └── components.py     # Reusable UI components
├── .env                  # Environment variables (API keys)
├── .gitignore            # Git exclusion rules
├── README.md             # Project documentation
└── requirements.txt      # Python dependency list

```

## Technical Stack

* **Framework:** [LangChain](https://www.langchain.com/)
* **LLMs:** Mistral AI (`mistral-large-latest`) & OpenAI (`gpt-4.1`) (HuggingFace)
* **Vector Database:** [ChromaDB](https://www.trychroma.com/) (Persistante)
* **Embeddings:** `Qwen/Qwen3-Embedding-8B` (HuggingFace)
* **UI:** [Streamlit](https://streamlit.io/)

## Performance Disclaimer

**Note on First Run Execution:** The first time you launch the application, the system needs to build the vector database. Depending on your hardware and the volume of documents in `rag_sources/`, this initial indexing may take several minutes. However, thanks to **ChromaDB persistence**, this process occurs only once. Subsequent launches will load the database instantly from the `chroma_db_cache/` folder.

## Installation & Setup

1. **Clone the repository**
```bash
git clone https://github.com/Thomas-Chambon/cardmaster.git
cd cardmaster
```
2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Environment Variables**

Create a `.env` file in the root directory with the following keys:

```
MISTRALAI_API_KEY=your-mistral-key-here
OPENAI_API_KEY=your-openai-key-here
TAVILY_API_KEY=your-tavily-key-here  
USER_AGENT=CardMasterAI/1.0      (example)
LANGSMITH_ENDPOINT=your-langsmith-endpoint-here  (example: https://eu.api.smith.langchain.com)
LANGSMITH_API_KEY=your-langsmith-key-here
LANGSMITH_PROJECT=your-langsmith-project-name
HUGGINGFACE_API_KEY=your-huggingface-key-here
HASH_CONFIG_FILE= (Set during first run)
```

Please visit the following sites to obtain your API keys:

- [Mistral AI](https://console.mistral.ai/api-keys/)
- [OpenAI](https://platform.openai.com/api-keys)
- [Tavily](https://app.tavily.com/keys) 
- [LangSmith (LangChain)](https://smith.langchain.com/settings)
- [Hugging Face](https://huggingface.co/settings/tokens)

## Usage

To launch the user interface:

```bash
streamlit run ui/app.py
```

## Adding sources to the RAG

CardMaster AI features a dynamic indexing system. You can expand the knowledge base by modifying the `rag_sources/config.json` file. The system will automatically detect these changes using SHA-256 hashing and update the vector store accordingly.

1. Supported Source Types

    Local PDFs: Place your files in the `rag_sources/` directory.

    Web URLs: Provide direct links to official rulebooks, card databases, or strategy articles.

2. Automatic Synchronization

Once you save the `config.json` file:

The RAG Engine detects a mismatch between the current file hash and the `HASH_CONFIG_FILE` stored in your `.env`.

It triggers the `document_loader.py` to fetch new content.

New deterministic IDs are generated for the chunks to ensure an "upsert" (update or insert) operation in ChromaDB, preventing any data duplication.

## License

This project is licensed under the [**GNU GPLv3 License**](LICENCE).

---

Developed by Thomas Chambon | 2025 CardMaster AI V.1 - Advanced RAG Architecture
