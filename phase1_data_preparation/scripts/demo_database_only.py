"""
Database Demo (No Dependencies Required)
Shows database queries without needing sentence-transformers

This script demonstrates the Phase 1 database functionality
without requiring any external packages (just built-in sqlite3)
"""

import sqlite3
import sys
from pathlib import Path


def demo_database_queries(db_path='../outputs/papers.db'):
    """Demonstrate structured database queries"""
    print("\n" + "="*60)
    print("DATABASE QUERIES (Phase 1 Demo)")
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
    
    # Query 5: Statistics
    print("Query 5: Database Statistics")
    print("-" * 40)
    cursor.execute("SELECT COUNT(*) FROM papers")
    paper_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM datasets")
    dataset_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM robots")
    robot_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM hardware WHERE type='compute'")
    compute_count = cursor.fetchone()[0]
    
    print(f"  • Total papers: {paper_count}")
    print(f"  • Unique datasets: {dataset_count}")
    print(f"  • Robot platforms: {robot_count}")
    print(f"  • Compute hardware types: {compute_count}")
    print()
    
    conn.close()


def main():
    """Run database demo"""
    print("\n" + "🔍"*30)
    print(" "*15 + "PHASE 1 DATABASE DEMO")
    print("🔍"*30)
    
    try:
        demo_database_queries()
        
        print("\n" + "="*60)
        print("✓ Database demo complete!")
        print("\nNote: This is a simplified demo showing only database queries.")
        print("For the full demo with semantic search, you need to:")
        print("  1. Fix NumPy version: bash ../fix_dependencies.sh")
        print("  2. Run: python demo_queries.py")
        print("="*60 + "\n")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: Database file not found - {e}")
        print("\nMake sure you're running from the scripts/ directory:")
        print("  cd phase1_data_preparation/scripts/")
        print("  python demo_database_only.py")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

