"""Document loading utilities."""
import os
import bs4
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader, PyMuPDFLoader, JSONLoader, CSVLoader

from .config import load_sources_config, RAG_SOURCES_DIR, get_logger

logger = get_logger(__name__)

class DocumentLoader:
    """Handle loading of web and PDF documents."""
    
    def __init__(self):
        self.config = load_sources_config()
    
    def load_web_documents(self) -> List[Document]:
        """Load documents from web URLs.
        
        Returns:
            List of loaded web documents.
        """
        urls = self.config.get('urls', {})
        all_urls = []
        
        # Combine all URLs from different sources
        for game_urls in urls.values():
            all_urls.extend(game_urls)
        
        if not all_urls:
            return []
        
        bs4_strainer = bs4.SoupStrainer(
            class_=("post-title", "post-header", "post-content")
        )
        
        try:
            loader = WebBaseLoader(
                web_paths=tuple(all_urls),
                bs_kwargs={"parse_only": bs4_strainer},
            )
            return loader.load()
        except Exception as e:
            logger.info(f"Error loading web documents: {e}")
            return []
    
    def load_pdf_documents(self) -> List[Document]:
        """Load documents from PDF files.
        
        Returns:
            List of loaded PDF documents.
        """
        pdf_paths = self.config.get('pdfs', [])
        docs_pdf = []
        
        for pdf_path in pdf_paths:
            full_path = RAG_SOURCES_DIR / pdf_path if not os.path.isabs(pdf_path) else pdf_path
            
            if os.path.exists(full_path):
                try:
                    loader = PyMuPDFLoader(str(full_path))
                    docs_pdf.extend(loader.load())
                except Exception as e:
                    logger.info(f"Error loading PDF {full_path}: {e}")
            else:
                logger.info(f"PDF not found: {full_path}")
        
        return docs_pdf
    
    def load_json_documents(self) -> List[Document]:
        """Load documents from JSON files.
    
        Returns:
            List of loaded JSON documents.
        """

        json_paths = self.config.get('jsons', [])
        docs_json = []

        for json_path in json_paths:
            full_path = RAG_SOURCES_DIR / json_path if not os.path.isabs(json_path) else json_path

            if os.path.exists(full_path):
                try:
                   loader = JSONLoader(
                        file_path=str(full_path),
                        jq_schema=".data | to_entries[]",
                        content_key="value",
                        text_content=False,
                    )
                   docs_json.extend(loader.load())
                except Exception as e:
                    logger.info(f"Error loading JSON {full_path}: {e}")
                
            else:
                logger.info(f"JSON not found {full_path}")

        return docs_json
    
    # def load_csv_documents(self) -> List[Document]:
    #     """Load documents from CSV files.
    
    #     Returns:
    #         List of loaded CSV documents.
    #     """
    #     csv_paths = self.config.get('csv', [])
    #     docs_csv = []

    #     for csv_path in csv_paths:
    #         full_path = RAG_SOURCES_DIR / csv_path if not os.path.isabs(csv_path) else csv_path

    #         if os.path.exists(full_path):
    #             try:
    #                 loader = CSVLoader(
    #                     file_path=str(full_path),
    #                     csv_args={
    #                         "delimiter": ",",
    #                         "quotechar": '"',
    #                         "fieldnames": ,
    #                     }
    #                 )
    #                 docs_csv.extend(loader.load())
    #             except Exception as e:
    #                 logger.info(f"Error loading CSV {full_path}: {e}")
                
    #         else:
    #             logger.info(f"CSV not found {full_path}")

    #     return docs_csv

    def load_all_documents(self) -> List[Document]:
        """Load all documents (web + PDF + JSON).
        
        Returns:
            List of all loaded documents.
        """
        web_docs = self.load_web_documents()
        pdf_docs = self.load_pdf_documents()
        json_docs = self.load_json_documents()
        
        logger.info(f"Loaded {len(web_docs)} web sources, {len(pdf_docs)} PDF sources, {len(json_docs)} JSON sources")
        
        return web_docs + pdf_docs + json_docs