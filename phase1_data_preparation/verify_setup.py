#!/usr/bin/env python3
"""
Quick verification script to check Phase 1 setup
Run this to verify all files and data are in place
"""

import json
import sqlite3
import csv
from pathlib import Path


def check_file_exists(path, description):
    """Check if a file exists and report"""
    if Path(path).exists():
        size = Path(path).stat().st_size
        print(f"  ✓ {description}: {path} ({size:,} bytes)")
        return True
    else:
        print(f"  ✗ {description}: {path} NOT FOUND")
        return False


def main():
    print("\n" + "="*60)
    print("Phase 1 Setup Verification")
    print("="*60 + "\n")
    
    all_ok = True
    
    # Check directory structure
    print("1. Directory Structure")
    print("-" * 40)
    dirs = ['data', 'data/full_text', 'outputs', 'scripts']
    for d in dirs:
        if Path(d).exists():
            print(f"  ✓ {d}/")
        else:
            print(f"  ✗ {d}/ NOT FOUND")
            all_ok = False
    print()
    
    # Check input data files
    print("2. Input Data Files")
    print("-" * 40)
    all_ok &= check_file_exists('data/metadata.json', 'Metadata')
    
    # Count text files
    txt_files = list(Path('data/full_text').glob('*.txt'))
    print(f"  ✓ Found {len(txt_files)} paper text files")
    print()
    
    # Check output files
    print("3. Output Files")
    print("-" * 40)
    all_ok &= check_file_exists('outputs/extracted_info.csv', 'Extracted CSV')
    all_ok &= check_file_exists('outputs/papers.db', 'SQLite Database')
    all_ok &= check_file_exists('outputs/faiss_index.bin', 'FAISS Index')
    all_ok &= check_file_exists('outputs/chunk_metadata.json', 'Chunk Metadata')
    print()
    
    # Check script files
    print("4. Script Files")
    print("-" * 40)
    scripts = [
        'scripts/1_extract_structured_info.py',
        'scripts/2_populate_database.py',
        'scripts/3_build_vector_index.py',
        'scripts/demo_queries.py'
    ]
    for script in scripts:
        all_ok &= check_file_exists(script, Path(script).name)
    print()
    
    # Validate data content
    print("5. Data Validation")
    print("-" * 40)
    
    # Check metadata
    try:
        with open('data/metadata.json', 'r') as f:
            metadata = json.load(f)
        print(f"  ✓ Metadata: {len(metadata)} papers")
    except Exception as e:
        print(f"  ✗ Metadata error: {e}")
        all_ok = False
    
    # Check CSV
    try:
        with open('outputs/extracted_info.csv', 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        print(f"  ✓ CSV: {len(rows)} papers extracted")
    except Exception as e:
        print(f"  ✗ CSV error: {e}")
        all_ok = False
    
    # Check database
    try:
        conn = sqlite3.connect('outputs/papers.db')
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM papers")
        paper_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM datasets")
        dataset_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM robots")
        robot_count = cursor.fetchone()[0]
        
        print(f"  ✓ Database: {paper_count} papers, {dataset_count} datasets, {robot_count} robots")
        conn.close()
    except Exception as e:
        print(f"  ✗ Database error: {e}")
        all_ok = False
    
    # Check chunk metadata
    try:
        with open('outputs/chunk_metadata.json', 'r') as f:
            chunk_data = json.load(f)
        chunk_count = chunk_data.get('num_chunks', 0)
        print(f"  ✓ Vector Index: {chunk_count} chunks indexed")
    except Exception as e:
        print(f"  ✗ Chunk metadata error: {e}")
        all_ok = False
    
    print()
    
    # Final summary
    print("="*60)
    if all_ok:
        print("✓ All checks passed! Phase 1 is ready for submission.")
        print("\nNext steps:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Run demo: cd scripts && python demo_queries.py")
        print("  3. Or run individual scripts to regenerate outputs")
    else:
        print("✗ Some checks failed. Please review the errors above.")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

