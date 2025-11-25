"""
Phase 1.4: Vector Index Creation for Semantic Search
Builds a FAISS index from paper text for RAG applications

Usage:
    python 3_build_vector_index.py [data_dir] [output_dir]
    
    data_dir: Directory containing full_text/ files (default: ../data)
    output_dir: Directory for output files (default: ../outputs)
    
Outputs:
    - faiss_index.bin: FAISS vector index
    - chunk_metadata.json: Chunk text and paper mappings
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass

from sentence_transformers import SentenceTransformer
import faiss


@dataclass
class Chunk:
    """Container for a chunk of text from a paper"""
    chunk_id: int
    paper_id: str
    text: str
    start_pos: int  # word index where chunk starts
    end_pos: int    # word index where chunk ends


class TextChunker:
    """
    Splits long paper text into smaller overlapping chunks.
    Overlap prevents loss of context at chunk boundaries.
    """
    
    def __init__(self, chunk_size=512, overlap=50):
        """
        Args:
            chunk_size: Number of words per chunk
            overlap: Number of words to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        print(f"TextChunker: {chunk_size} words/chunk, {overlap} word overlap")
    
    def chunk_text(self, text, paper_id, start_chunk_id=0):
        """
        Break text into overlapping chunks.
        
        Args:
            text: Full paper text
            paper_id: Identifier for the paper
            start_chunk_id: Starting ID (useful for multiple papers)
            
        Returns:
            List of Chunk objects
        """
        words = text.split()
        chunks = []
        chunk_id = start_chunk_id
        stride = self.chunk_size - self.overlap
        
        i = 0
        while i < len(words):
            end = min(i + self.chunk_size, len(words))
            chunk_words = words[i:end]
            chunk_text = ' '.join(chunk_words)
            
            chunk = Chunk(
                chunk_id=chunk_id,
                paper_id=paper_id,
                text=chunk_text,
                start_pos=i,
                end_pos=end
            )
            chunks.append(chunk)
            
            chunk_id += 1
            i += stride
            
            if end >= len(words):
                break
        
        return chunks


class VectorIndexBuilder:
    """
    Builds and searches a FAISS index for semantic similarity.
    Handles embedding generation using sentence-transformers.
    """
    
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """
        Args:
            model_name: Sentence transformer model to use
                       (all-MiniLM-L6-v2 is fast and effective)
        """
        self.model_name = model_name
        self.model = None
        self.index = None
        self.chunks = []
        
        print(f"VectorIndexBuilder using model: {model_name}")
    
    def load_model(self):
        """Load the sentence transformer model"""
        print(f"Loading model: {self.model_name}...")
        self.model = SentenceTransformer(self.model_name)
        embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"  ✓ Model loaded! Embedding dim: {embedding_dim}")
    
    def create_embeddings(self, texts):
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of strings to embed
            
        Returns:
            Numpy array of shape (num_texts, embedding_dim)
        """
        print(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings
    
    def build_index(self, chunks):
        """
        Build the FAISS index from chunks.
        
        Args:
            chunks: List of Chunk objects
        """
        if not self.model:
            print("ERROR: Model not loaded! Call load_model() first")
            return
        
        if len(chunks) == 0:
            print("ERROR: No chunks to index!")
            return
        
        print(f"\nBuilding FAISS index from {len(chunks)} chunks...")
        
        # Extract text from all chunks
        texts = [chunk.text for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.create_embeddings(texts)
        
        # Get embedding dimension
        dim = embeddings.shape[1]
        print(f"Embedding dimension: {dim}")
        
        # Create FAISS index (Flat L2 for exact search)
        self.index = faiss.IndexFlatL2(dim)
        
        # Add embeddings to index
        self.index.add(embeddings.astype('float32'))
        
        # Save chunks for retrieval
        self.chunks = chunks
        
        print(f"  ✓ Index built! Total vectors: {self.index.ntotal}")
    
    def save_index(self, index_path="faiss_index.bin", metadata_path="chunk_metadata.json"):
        """
        Save index and metadata to disk for reuse.
        
        Args:
            index_path: Path for FAISS index file
            metadata_path: Path for chunk metadata JSON
        """
        # Ensure output directory exists
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving index to {index_path}...")
        faiss.write_index(self.index, index_path)
        
        print(f"Saving metadata to {metadata_path}...")
        metadata = {
            'model_name': self.model_name,
            'num_chunks': len(self.chunks),
            'chunks': [
                {
                    'chunk_id': c.chunk_id,
                    'paper_id': c.paper_id,
                    'text': c.text,
                    'start_pos': c.start_pos,
                    'end_pos': c.end_pos
                }
                for c in self.chunks
            ]
        }
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        print("  ✓ Saved!")
    
    def load_index(self, index_path="faiss_index.bin", metadata_path="chunk_metadata.json"):
        """Load a previously saved index"""
        print(f"Loading index from {index_path}...")
        self.index = faiss.read_index(index_path)
        print(f"  ✓ Loaded index with {self.index.ntotal} vectors")
        
        print(f"Loading metadata from {metadata_path}...")
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Reconstruct Chunk objects
        self.chunks = [
            Chunk(
                chunk_id=c['chunk_id'],
                paper_id=c['paper_id'],
                text=c['text'],
                start_pos=c['start_pos'],
                end_pos=c['end_pos']
            )
            for c in metadata['chunks']
        ]
        
        self.model_name = metadata['model_name']
        print(f"  ✓ Loaded {len(self.chunks)} chunks (model: {self.model_name})")
    
    def search(self, query, top_k=5):
        """
        Search for most similar chunks to a query.
        
        Args:
            query: Text string to search for
            top_k: Number of results to return
            
        Returns:
            List of (Chunk, distance_score) tuples sorted by relevance
        """
        if not self.model:
            print("ERROR: Need to load model first!")
            return []
        
        # Embed the query
        query_embedding = self.model.encode([query])
        
        # Search the index
        distances, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        # Retrieve chunks and pair with scores
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            chunk = self.chunks[idx]
            # Lower distance = more similar (L2 distance)
            results.append((chunk, float(dist)))
        
        return results


def load_all_papers(papers_dir):
    """
    Load all paper texts from directory.
    
    Args:
        papers_dir: Directory containing .txt files
        
    Returns:
        Dict of {paper_id: text}
    """
    papers = {}
    papers_path = Path(papers_dir)
    
    if not papers_path.exists():
        print(f"ERROR: Directory not found: {papers_dir}")
        return papers
    
    txt_files = list(papers_path.glob("*.txt"))
    print(f"Found {len(txt_files)} paper files")
    
    for txt_file in txt_files:
        paper_id = txt_file.stem
        
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                text = f.read()
                papers[paper_id] = text
                print(f"  ✓ Loaded {paper_id} ({len(text)} chars)")
        except Exception as e:
            print(f"  ✗ Error loading {paper_id}: {e}")
    
    return papers


def main():
    """Main pipeline for creating the vector index"""
    import sys
    
    print("\n" + "="*60)
    print("Phase 1.4: Vector Index Creation")
    print("="*60 + "\n")
    
    # Parse arguments
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "../data"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "../outputs"
    
    # Step 1: Load all papers
    print("Step 1: Loading papers...")
    papers_dir = Path(data_dir) / "full_text"
    papers = load_all_papers(papers_dir)
    print(f"  ✓ Loaded {len(papers)} papers\n")
    
    # Step 2: Chunk all papers
    print("Step 2: Chunking papers...")
    chunker = TextChunker(chunk_size=512, overlap=50)
    all_chunks = []
    chunk_id = 0
    
    for paper_id, text in papers.items():
        chunks = chunker.chunk_text(text, paper_id, chunk_id)
        all_chunks.extend(chunks)
        chunk_id += len(chunks)
        print(f"  {paper_id}: {len(chunks)} chunks")
    
    print(f"\n  ✓ Created {len(all_chunks)} chunks total\n")
    
    # Step 3: Build index
    print("Step 3: Building FAISS index...")
    builder = VectorIndexBuilder()
    builder.load_model()
    builder.build_index(all_chunks)
    
    # Step 4: Save to disk
    print("\nStep 4: Saving index...")
    index_path = str(Path(output_dir) / "faiss_index.bin")
    metadata_path = str(Path(output_dir) / "chunk_metadata.json")
    builder.save_index(index_path, metadata_path)
    
    print("\n" + "="*60)
    print(f"✓ Index saved to: {index_path}")
    print(f"✓ Metadata saved to: {metadata_path}")
    print("="*60)
    
    # Quick test search
    print("\n🔍 Testing search with query: 'vision-language-action models'")
    results = builder.search("vision-language-action models for robotics", top_k=3)
    for i, (chunk, score) in enumerate(results, 1):
        print(f"\n{i}. Paper: {chunk.paper_id} (distance: {score:.2f})")
        print(f"   Preview: {chunk.text[:150]}...")
    
    print()


if __name__ == "__main__":
    main()

