"""
Retrieval module for fetching relevant context from Wikipedia.
"""
import wikipedia
import re
from typing import List, Dict, Tuple, Optional
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WikipediaRetriever:
    """
    Retriever that fetches relevant paragraphs from Wikipedia.
    """
    def __init__(self, language: str = "en", max_pages: int = 5, max_paragraphs: int = 10):
        """
        Initialize the Wikipedia retriever.
        
        Args:
            language: Wikipedia language
            max_pages: Maximum number of pages to fetch
            max_paragraphs: Maximum number of paragraphs to return
        """
        self.language = language
        self.max_pages = max_pages
        self.max_paragraphs = max_paragraphs
        wikipedia.set_lang(language)
        self.vectorizer = TfidfVectorizer(stop_words='english')
        
    def search(self, query: str) -> List[Dict]:
        """
        Search Wikipedia for relevant pages and extract paragraphs.
        
        Args:
            query: Search query
            
        Returns:
            List of dictionaries with page titles and paragraphs
        """
        logger.info(f"Searching Wikipedia for: {query}")
        try:
            # Search for relevant pages
            search_results = wikipedia.search(query, results=self.max_pages)
            
            if not search_results:
                logger.warning(f"No Wikipedia results found for: {query}")
                return []
            
            results = []
            for title in search_results:
                try:
                    # Get page content
                    page = wikipedia.page(title, auto_suggest=False)
                    
                    # Extract paragraphs (non-empty lines)
                    paragraphs = [p for p in page.content.split('\n') if p.strip()]
                    
                    # Filter out short paragraphs and section headers
                    paragraphs = [p for p in paragraphs if len(p) > 50 and not p.startswith('==')]
                    
                    if paragraphs:
                        results.append({
                            "title": page.title,
                            "url": page.url,
                            "paragraphs": paragraphs
                        })
                except (wikipedia.exceptions.DisambiguationError, 
                        wikipedia.exceptions.PageError) as e:
                    logger.warning(f"Error fetching Wikipedia page '{title}': {e}")
                    continue
            
            return results
        except Exception as e:
            logger.error(f"Error searching Wikipedia: {e}")
            return []
    
    def get_relevant_paragraphs(self, query: str) -> List[Dict]:
        """
        Get paragraphs relevant to the query.
        
        Args:
            query: Search query
            
        Returns:
            List of dictionaries with page titles, URLs, and relevant paragraphs
        """
        # Get search results
        search_results = self.search(query)
        
        if not search_results:
            return []
        
        # Extract all paragraphs
        all_paragraphs = []
        paragraph_sources = []
        
        for result in search_results:
            for paragraph in result["paragraphs"]:
                all_paragraphs.append(paragraph)
                paragraph_sources.append({
                    "title": result["title"],
                    "url": result["url"],
                    "paragraph": paragraph
                })
        
        # Rank paragraphs by relevance to query
        ranked_paragraphs = self._rank_paragraphs(query, all_paragraphs)
        
        # Get top paragraphs
        top_paragraphs = []
        for idx in ranked_paragraphs[:self.max_paragraphs]:
            source = paragraph_sources[idx]
            top_paragraphs.append({
                "title": source["title"],
                "url": source["url"],
                "text": source["paragraph"],
                "score": float(ranked_paragraphs[idx][1])  # Convert numpy float to Python float
            })
        
        return top_paragraphs
    
    def _rank_paragraphs(self, query: str, paragraphs: List[str]) -> List[Tuple[int, float]]:
        """
        Rank paragraphs by relevance to query using TF-IDF and cosine similarity.
        
        Args:
            query: Search query
            paragraphs: List of paragraphs
            
        Returns:
            List of tuples with paragraph index and relevance score
        """
        if not paragraphs:
            return []
        
        try:
            # Create document matrix with query as first document
            documents = [query] + paragraphs
            tfidf_matrix = self.vectorizer.fit_transform(documents)
            
            # Calculate cosine similarity between query and paragraphs
            query_vector = tfidf_matrix[0:1]
            paragraph_vectors = tfidf_matrix[1:]
            similarities = cosine_similarity(query_vector, paragraph_vectors)[0]
            
            # Rank paragraphs by similarity
            ranked = [(i, similarities[i]) for i in range(len(similarities))]
            ranked.sort(key=lambda x: x[1], reverse=True)
            
            return ranked
        except Exception as e:
            logger.error(f"Error ranking paragraphs: {e}")
            # Return paragraphs in original order with zero scores
            return [(i, 0.0) for i in range(len(paragraphs))]

class DenseRetriever:
    """
    Dense retriever using sentence transformers and FAISS.
    This is a placeholder for future implementation.
    """
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the dense retriever.
        
        Args:
            model_name: Sentence transformer model name
        """
        try:
            from sentence_transformers import SentenceTransformer
            import faiss
            
            self.model = SentenceTransformer(model_name)
            self.faiss_available = True
            logger.info(f"Dense retriever initialized with model: {model_name}")
        except ImportError:
            logger.warning("sentence-transformers or faiss not available. Dense retrieval disabled.")
            self.faiss_available = False
    
    def index_documents(self, documents: List[str]) -> Optional[np.ndarray]:
        """
        Create FAISS index for documents.
        
        Args:
            documents: List of documents to index
            
        Returns:
            Document embeddings
        """
        if not self.faiss_available:
            return None
        
        # This is a placeholder for actual implementation
        # In a real implementation, you would:
        # 1. Encode documents using the sentence transformer
        # 2. Create a FAISS index
        # 3. Add document embeddings to the index
        # 4. Save the index for future use
        
        return None
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve relevant documents for query.
        
        Args:
            query: Search query
            top_k: Number of documents to retrieve
            
        Returns:
            List of dictionaries with document text and score
        """
        if not self.faiss_available:
            return []
        
        # This is a placeholder for actual implementation
        # In a real implementation, you would:
        # 1. Encode the query using the sentence transformer
        # 2. Search the FAISS index for similar documents
        # 3. Return the top-k documents with scores
        
        return []

def get_context_for_question(question: str, use_dense_retrieval: bool = False) -> str:
    """
    Get context for a question from Wikipedia.
    
    Args:
        question: Question text
        use_dense_retrieval: Whether to use dense retrieval
        
    Returns:
        Context text
    """
    retriever = WikipediaRetriever()
    paragraphs = retriever.get_relevant_paragraphs(question)
    
    if not paragraphs:
        return ""
    
    # Return the most relevant paragraph
    return paragraphs[0]["text"]

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Retrieve context from Wikipedia")
    parser.add_argument("--query", required=True, help="Search query")
    parser.add_argument("--max_pages", type=int, default=5, help="Maximum number of pages to fetch")
    parser.add_argument("--max_paragraphs", type=int, default=3, help="Maximum number of paragraphs to return")
    
    args = parser.parse_args()
    
    retriever = WikipediaRetriever(max_pages=args.max_pages, max_paragraphs=args.max_paragraphs)
    paragraphs = retriever.get_relevant_paragraphs(args.query)
    
    print(f"Found {len(paragraphs)} relevant paragraphs for query: {args.query}\n")
    
    for i, p in enumerate(paragraphs):
        print(f"Paragraph {i+1} (Score: {p['score']:.4f}):")
        print(f"Source: {p['title']} ({p['url']})")
        print(f"Text: {p['text'][:200]}...\n")