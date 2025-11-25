# Phase 1: Data Preparation Pipeline

This directory contains the complete data preparation pipeline for building a Retrieval-Augmented Generation (RAG) system for robotics research papers. The pipeline extracts structured information from papers, populates a queryable database, and creates a semantic search index.

## 📁 Directory Structure

```
phase1_data_preparation/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── data/                             # Input data
│   ├── full_text/                    # Extracted paper text (18 .txt files)
│   └── metadata.json                 # Paper metadata (title, year, venue)
├── scripts/
│   ├── 1_extract_structured_info.py  # Step 1: Extract structured data
│   ├── 2_populate_database.py        # Step 2: Create SQLite database
│   ├── 3_build_vector_index.py       # Step 3: Build FAISS index
│   └── demo_queries.py               # Example usage
└── outputs/                          # Generated outputs (already included)
    ├── extracted_info.csv            # Structured data from papers
    ├── papers.db                     # SQLite database
    ├── faiss_index.bin               # FAISS vector index
    └── chunk_metadata.json           # Chunk text and mappings
```

## 🚀 Quick Start

### 1. Install Dependencies

**Recommended: Use a fresh environment**

```bash
# Option A: Using conda (recommended)
conda create -n phase1 python=3.10
conda activate phase1
pip install -r requirements.txt

# Option B: Using venv
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**If you have dependency conflicts:**

```bash
bash fix_dependencies.sh
```

**Note**: The scripts use standard Python libraries (`sqlite3`, `csv`, `json`, `re`, `pathlib`) which are built-in. External dependencies:
- `sentence-transformers==2.5.1` - For text embeddings
- `faiss-cpu==1.7.4` - For vector similarity search
- `numpy==1.24.3` - For numerical operations
- `torch>=2.0.0` - PyTorch backend
- `transformers==4.37.2` - Transformer models

### 2. Run the Pipeline

The pipeline consists of 3 sequential steps. Each script can be run independently with default paths:

```bash
cd scripts/

# Step 1: Extract structured information from papers
python 1_extract_structured_info.py

# Step 2: Populate SQLite database from CSV
python 2_populate_database.py

# Step 3: Build FAISS vector index
python 3_build_vector_index.py
```

### 3. Run Demo Queries

```bash
python demo_queries.py
```

This will demonstrate:
- Database queries (e.g., "Find papers using BridgeData V2")
- Semantic search (e.g., "What papers discuss sim-to-real transfer?")

## 📊 Pipeline Overview

### Step 1: Structured Information Extraction

**Script**: `1_extract_structured_info.py`

Extracts structured data from paper text using regex pattern matching:
- **Metadata**: Title, year, venue
- **Datasets**: Training/evaluation datasets (BridgeData, DROID, Open-X Embodiment, etc.)
- **Models**: Model names, architectures, sizes (RT-1, OpenVLA, Octo, etc.)
- **Hardware**: Robot platforms (Franka Panda, WidowX), compute (A100, TPU)
- **Training**: Optimizer, learning rate, batch size, epochs
- **Evaluation**: Tasks, success rates, baselines

**Input**: `data/full_text/*.txt` + `data/metadata.json`  
**Output**: `outputs/extracted_info.csv` (26 fields)

**Usage**:
```bash
python 1_extract_structured_info.py [data_dir] [output_csv]
```

### Step 2: Database Population

**Script**: `2_populate_database.py`

Creates a normalized SQLite database from the extracted CSV:

**Database Schema** (11 tables):
- `papers` - Main paper information (20 fields)
- `datasets` - Unique datasets
- `robots` - Robot platforms
- `hardware` - Sensors, grippers, GPUs (with type: 'robot' or 'compute')
- `baselines` - Baseline models for comparison
- `tasks` - Evaluation tasks
- `paper_datasets`, `paper_robots`, `paper_hardware`, `paper_baselines` - Junction tables

**Input**: `outputs/extracted_info.csv`  
**Output**: `outputs/papers.db`

**Usage**:
```bash
python 2_populate_database.py [csv_path] [db_path]
```

**Example Queries**:
```sql
-- Find papers using BridgeData V2
SELECT p.title, p.year FROM papers p
JOIN paper_datasets pd ON p.paper_id = pd.paper_id
JOIN datasets d ON pd.dataset_id = d.id
WHERE d.name = 'BridgeData V2';

-- Papers with model sizes
SELECT paper_id, model_name, model_size 
FROM papers 
WHERE model_size IS NOT NULL;

-- Most common datasets
SELECT d.name, COUNT(*) as count 
FROM datasets d
JOIN paper_datasets pd ON d.id = pd.dataset_id
GROUP BY d.name
ORDER BY count DESC;
```

### Step 3: Vector Index Creation

**Script**: `3_build_vector_index.py`

Builds a FAISS index for semantic similarity search:

**Process**:
1. **Chunking**: Splits papers into 512-word chunks with 50-word overlap
2. **Embedding**: Generates 384-dim embeddings using `all-MiniLM-L6-v2`
3. **Indexing**: Creates FAISS Flat L2 index for exact similarity search
4. **Saving**: Stores index and chunk metadata for reuse

**Input**: `data/full_text/*.txt`  
**Output**: `outputs/faiss_index.bin` + `outputs/chunk_metadata.json`

**Usage**:
```bash
python 3_build_vector_index.py [data_dir] [output_dir]
```

**Statistics** (from 18 papers):
- **Chunks**: ~406 chunks total
- **Embedding time**: ~4 seconds
- **Search time**: < 0.1 seconds per query
- **Index size**: Manageable in memory

## 🔍 Usage Examples

### Database Queries

```python
import sqlite3

conn = sqlite3.connect('outputs/papers.db')
cursor = conn.cursor()

# Find papers using Franka Panda robot
cursor.execute("""
    SELECT p.title FROM papers p
    JOIN paper_robots pr ON p.paper_id = pr.paper_id
    JOIN robots r ON pr.robot_id = r.id
    WHERE r.name = 'Franka Panda'
""")
print(cursor.fetchall())
```

### Semantic Search

```python
from scripts.build_vector_index import VectorIndexBuilder
from sentence_transformers import SentenceTransformer

# Load index
builder = VectorIndexBuilder()
builder.model = SentenceTransformer('all-MiniLM-L6-v2')
builder.load_index('outputs/faiss_index.bin', 'outputs/chunk_metadata.json')

# Search
results = builder.search("vision-language-action models", top_k=3)
for chunk, score in results:
    print(f"Paper: {chunk.paper_id}, Score: {score:.2f}")
    print(f"Text: {chunk.text[:150]}...\n")
```

## 📈 Statistics

From 18 robotics papers (2017-2025):

| Metric | Count |
|--------|-------|
| Papers processed | 18 |
| Text extracted | 1.6M+ characters |
| Chunks created | ~406 |
| Unique datasets | 11 |
| Robot platforms | 5 |
| Baseline models | 7 |
| Database tables | 11 |
| Search latency | < 0.1s |

## 🛠️ Technical Details

### Extraction Patterns

The extraction script uses regex patterns to identify:
- **15 datasets**: Open-X Embodiment, BridgeData, DROID, etc.
- **10 robot platforms**: Franka Panda, WidowX, Unitree H1, etc.
- **7 compute types**: A100, V100, RTX 4090, TPU, etc.
- **7 vision encoders**: DINOv2, CLIP, SigLIP, ViT, etc.
- **7 VLA models**: RT-1, RT-2, OpenVLA, Octo, etc.
- **7 simulators**: Isaac Sim, MuJoCo, Habitat, etc.

### Database Design

- **Normalized schema**: Eliminates redundancy via junction tables
- **Indexed columns**: Fast lookups on `paper_id`, `dataset_name`, `robot_name`
- **Type classification**: Hardware categorized as 'robot' or 'compute'

### Vector Index

- **Model**: `all-MiniLM-L6-v2` (384-dim embeddings)
- **Index type**: FAISS Flat L2 (exact nearest neighbor search)
- **Chunking strategy**: 512 words with 50-word overlap to preserve context
- **Storage**: Binary index + JSON metadata for easy loading

## 🧪 Testing

All outputs are pre-generated and included in the `outputs/` directory. To verify:

```bash
# Check CSV
head outputs/extracted_info.csv

# Check database
sqlite3 outputs/papers.db "SELECT COUNT(*) FROM papers;"

# Check index (in Python)
python -c "import faiss; idx = faiss.read_index('outputs/faiss_index.bin'); print(f'Vectors: {idx.ntotal}')"
```

## 📝 Notes for TAs

### What's Included

✅ **Input Data**: Full text files (18 papers) and metadata  
✅ **Scripts**: 3 pipeline scripts with clear numbering and documentation  
✅ **Outputs**: Pre-generated CSV, database, and vector index  
✅ **Demo**: Example query script showing usage  
✅ **Documentation**: This comprehensive README  

### What's NOT Included

❌ **Original PDFs**: Not necessary - we provide the extracted text  
❌ **Python Virtual Environment**: Install dependencies via `requirements.txt`  
❌ **Intermediate Files**: Only final outputs are included  

### Expected Runtime

Running the full pipeline from scratch:
- **Step 1**: ~5 seconds (extraction)
- **Step 2**: ~2 seconds (database)
- **Step 3**: ~8 seconds (indexing with embeddings)
- **Total**: < 20 seconds on a standard laptop

### System Requirements

- **Python**: 3.8+
- **RAM**: 4GB+ (for FAISS index)
- **Storage**: ~100MB for outputs
- **GPU**: Not required (CPU-only FAISS is sufficient)

## 🤝 Questions?

This pipeline was designed to be:
1. **Simple**: Clear step-by-step process
2. **Self-contained**: All dependencies explicit
3. **Reproducible**: Pre-generated outputs included
4. **Well-documented**: Comprehensive docstrings and README

For any issues, check:
- Dependencies are installed: `pip list | grep -E 'sentence-transformers|faiss|numpy'`
- Paths are correct: Scripts use relative paths from `scripts/` directory
- Python version: 3.8 or higher

---

**Author**: NLP Project - Thesaurus Virtual Assistant  
**Date**: November 2025  
**Phase**: 1 of 8 (Data Preparation)

