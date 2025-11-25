"""
Demo Script: Query the Database and Vector Index
Demonstrates how to use the outputs from Phase 1

This script shows:
1. Database queries (structured data)
2. Semantic search (vector similarity)
3. Combining both for comprehensive results
"""

import sqlite3
import sys
from pathlib import Path

# Import the vector indexing components
sys.path.insert(0, str(Path(__file__).parent))
from sentence_transformers import SentenceTransformer
import faiss
import json


def demo_database_queries(db_path='../outputs/papers.db'):
    """Demonstrate structured database queries"""
    print("\n" + "="*60)
    print("1. DATABASE QUERIES (Structured Data)")
    print("="*60 + "\n")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Query 1: Papers using specific dataset
    print("Query 1: Papers using BridgeData V2")
    print("-" * 40)
    cursor.execute("""
        SELECT p.title, p.year 
        FROM papers p
        JOIN paper_datasets pd ON p.paper_id = pd.paper_id
        JOIN datasets d ON pd.dataset_id = d.id
        WHERE d.name = 'BridgeData V2'
    """)
    results = cursor.fetchall()
    for title, year in results:
        print(f"  • [{year}] {title}")
    print()
    
    # Query 2: Papers with model sizes
    print("Query 2: Papers with reported model sizes")
    print("-" * 40)
    cursor.execute("""
        SELECT paper_id, model_name, model_size 
        FROM papers 
        WHERE model_size IS NOT NULL
    """)
    results = cursor.fetchall()
    for paper_id, model_name, model_size in results:
        print(f"  • {model_name or paper_id}: {model_size}")
    print()
    
    # Query 3: Most common datasets
    print("Query 3: Most common datasets across papers")
    print("-" * 40)
    cursor.execute("""
        SELECT d.name, COUNT(*) as count 
        FROM datasets d
        JOIN paper_datasets pd ON d.id = pd.dataset_id
        GROUP BY d.name
        ORDER BY count DESC
        LIMIT 5
    """)
    results = cursor.fetchall()
    for dataset, count in results:
        print(f"  • {dataset}: {count} papers")
    print()
    
    # Query 4: Papers by robot platform
    print("Query 4: Papers using Franka Panda robot")
    print("-" * 40)
    cursor.execute("""
        SELECT p.title, p.year
        FROM papers p
        JOIN paper_robots pr ON p.paper_id = pr.paper_id
        JOIN robots r ON pr.robot_id = r.id
        WHERE r.name = 'Franka Panda'
    """)
    results = cursor.fetchall()
    for title, year in results:
        print(f"  • [{year}] {title}")
    print()
    
    # Query 5: Training details
    print("Query 5: Papers with complete training details")
    print("-" * 40)
    cursor.execute("""
        SELECT paper_id, optimizer, learning_rate, batch_size
        FROM papers 
        WHERE optimizer IS NOT NULL 
          AND learning_rate IS NOT NULL
          AND batch_size IS NOT NULL
    """)
    results = cursor.fetchall()
    for paper_id, opt, lr, bs in results:
        print(f"  • {paper_id}: {opt}, LR={lr}, BS={bs}")
    print()
    
    conn.close()


def demo_semantic_search(
    index_path='../outputs/faiss_index.bin',
    metadata_path='../outputs/chunk_metadata.json'
):
    """Demonstrate semantic similarity search"""
    print("\n" + "="*60)
    print("2. SEMANTIC SEARCH (Vector Similarity)")
    print("="*60 + "\n")
    
    # Load the model
    print("Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Load the index
    print(f"Loading FAISS index from {index_path}...")
    index = faiss.read_index(index_path)
    
    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    chunks = metadata['chunks']
    
    print(f"✓ Loaded index with {index.ntotal} vectors\n")
    
    # Define test queries
    test_queries = [
        "vision-language-action models for robotics",
        "sim-to-real transfer learning",
        "data augmentation techniques",
        "OpenVLA architecture and training"
    ]
    
    for query_num, query in enumerate(test_queries, 1):
        print(f"Query {query_num}: '{query}'")
        print("-" * 40)
        
        # Encode query
        query_embedding = model.encode([query])
        
        # Search
        distances, indices = index.search(query_embedding.astype('float32'), k=3)
        
        # Display results
        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0]), 1):
            chunk = chunks[idx]
            paper_id = chunk['paper_id']
            text = chunk['text']
            
            # Show first 200 characters
            preview = text[:200] + "..." if len(text) > 200 else text
            
            print(f"\n  {rank}. Paper: {paper_id} (distance: {dist:.2f})")
            print(f"     Preview: {preview}")
        
        print()


def demo_combined_query(
    db_path='../outputs/papers.db',
    index_path='../outputs/faiss_index.bin',
    metadata_path='../outputs/chunk_metadata.json'
):
    """Demonstrate combining database and semantic search"""
    print("\n" + "="*60)
    print("3. COMBINED QUERY (Database + Semantic Search)")
    print("="*60 + "\n")
    
    query = "What datasets does OpenVLA use for training?"
    print(f"Question: {query}\n")
    
    # Step 1: Database query for structured info
    print("Step 1: Query database for OpenVLA training datasets")
    print("-" * 40)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT p.model_name, d.name, pd.dataset_type
        FROM papers p
        JOIN paper_datasets pd ON p.paper_id = pd.paper_id
        JOIN datasets d ON pd.dataset_id = d.id
        WHERE p.model_name LIKE '%OpenVLA%'
    """)
    results = cursor.fetchall()
    
    if results:
        print("Database results:")
        for model, dataset, dtype in results:
            print(f"  • {dataset} ({dtype})")
    else:
        print("  No direct database results")
    print()
    
    # Step 2: Semantic search for context
    print("Step 2: Semantic search for related information")
    print("-" * 40)
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    index = faiss.read_index(index_path)
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    chunks = metadata['chunks']
    
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding.astype('float32'), k=2)
    
    print("Semantic search results:")
    for dist, idx in zip(distances[0], indices[0]):
        chunk = chunks[idx]
        if 'openvla' in chunk['paper_id'].lower():
            preview = chunk['text'][:300] + "..."
            print(f"\n  From {chunk['paper_id']}:")
            print(f"  {preview}")
    
    conn.close()
    print()


def main():
    """Run all demo queries"""
    print("\n" + "🔍"*30)
    print(" "*20 + "PHASE 1 DEMO: Query System")
    print("🔍"*30)
    
    try:
        # Demo 1: Database queries
        demo_database_queries()
        
        # Demo 2: Semantic search
        demo_semantic_search()
        
        # Demo 3: Combined approach
        demo_combined_query()
        
        print("\n" + "="*60)
        print("✓ Demo complete! All queries executed successfully.")
        print("="*60 + "\n")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: Required file not found - {e}")
        print("\nMake sure you're running from the scripts/ directory:")
        print("  cd phase1_data_preparation/scripts/")
        print("  python demo_queries.py")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

