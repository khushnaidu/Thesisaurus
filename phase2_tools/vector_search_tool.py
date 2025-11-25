import json
from pathlib import Path


class VectorSearchTool:
    """Semantic search using FAISS vector index"""
    
    def __init__(
        self,
        index_path="../phase1_data_preparation/outputs/faiss_index.bin",
        metadata_path="../phase1_data_preparation/outputs/chunk_metadata.json"
    ):
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        
        if not self.index_path.exists():
            raise FileNotFoundError(f"Index not found: {index_path}")
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        
        self.model = None
        self.index = None
        self.chunks = None
    
    def _load_resources(self):
        """Lazy load model and index to save memory"""
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            import faiss
            
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.index = faiss.read_index(str(self.index_path))
            
            with open(self.metadata_path, 'r') as f:
                metadata = json.load(f)
                self.chunks = metadata['chunks']
    
    def search(self, query, top_k=5):
        """Semantic search across all papers"""
        try:
            self._load_resources()
            
            query_embedding = self.model.encode([query])
            distances, indices = self.index.search(query_embedding.astype('float32'), top_k)
            
            results = []
            for idx, dist in zip(indices[0], distances[0]):
                chunk = self.chunks[idx]
                results.append({
                    "paper_id": chunk['paper_id'],
                    "text": chunk['text'],
                    "score": float(dist),
                    "chunk_id": chunk['chunk_id']
                })
            
            return {
                "success": True,
                "query": query,
                "count": len(results),
                "results": results
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def search_within_paper(self, query, paper_id, top_k=3):
        """Search within a specific paper"""
        try:
            self._load_resources()
            
            paper_chunk_indices = [
                i for i, chunk in enumerate(self.chunks)
                if chunk['paper_id'] == paper_id
            ]
            
            if not paper_chunk_indices:
                return {"success": False, "error": f"No chunks found for paper {paper_id}"}
            
            query_embedding = self.model.encode([query])
            
            import numpy as np
            chunk_embeddings = np.array([
                self.index.reconstruct(i) for i in paper_chunk_indices
            ])
            
            from numpy.linalg import norm
            distances = [norm(query_embedding[0] - chunk_emb) for chunk_emb in chunk_embeddings]
            sorted_indices = np.argsort(distances)[:top_k]
            
            results = []
            for idx in sorted_indices:
                original_idx = paper_chunk_indices[idx]
                chunk = self.chunks[original_idx]
                results.append({
                    "paper_id": chunk['paper_id'],
                    "text": chunk['text'],
                    "score": float(distances[idx]),
                    "chunk_id": chunk['chunk_id']
                })
            
            return {
                "success": True,
                "query": query,
                "paper_id": paper_id,
                "count": len(results),
                "results": results
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_paper_chunks(self, paper_id):
        """Get all chunks for a paper"""
        try:
            self._load_resources()
            
            paper_chunks = [
                chunk for chunk in self.chunks
                if chunk['paper_id'] == paper_id
            ]
            
            if not paper_chunks:
                return {"success": False, "error": f"No chunks found for paper {paper_id}"}
            
            return {
                "success": True,
                "paper_id": paper_id,
                "count": len(paper_chunks),
                "chunks": paper_chunks
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
