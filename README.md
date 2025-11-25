# NLP Project: Thesaurus Virtual Assistant for Robotics Research

Building a Retrieval-Augmented Generation (RAG) system for analyzing robotics research papers.

## 📁 Project Structure

```
NLP_Project/
├── phase1_data_preparation/     # ✅ Phase 1: Complete data pipeline (READY)
│   ├── data/                    # Input: 18 papers (text + metadata)
│   ├── scripts/                 # Pipeline: extract → database → vector index
│   ├── outputs/                 # Generated: CSV, SQLite DB, FAISS index
│   ├── README.md               # Full documentation
│   └── SUBMISSION_GUIDE.md     # Quick start guide
├── papers/                      # Source: Raw PDF files (18 papers)
├── processed_papers/            # Source: Extracted text from PDFs
├── scripts/                     # Development utilities
│   ├── research_pdf_parser_clean.py  # PDF parsing (if needed for more papers)
│   ├── query_database.py             # Database query helper
│   └── thesisaurus_va.ipynb          # Development notebook
├── venv/                        # Python virtual environment
└── README.md                    # This file
```

## 🚀 Quick Start

### Phase 1: Data Preparation (Complete ✅)

All Phase 1 work is organized in the **`phase1_data_preparation/`** directory.

```bash
# Navigate to Phase 1
cd phase1_data_preparation

# Verify setup (< 30 seconds)
python verify_setup.py

# See full documentation
cat README.md

# Or jump straight to demo
cd scripts
python demo_queries.py
```

**What Phase 1 Includes:**
- ✅ Text extraction from 18 robotics papers
- ✅ Structured data extraction (26 fields per paper)
- ✅ SQLite database (11 normalized tables)
- ✅ FAISS vector index (406 chunks for semantic search)
- ✅ Complete documentation and demo scripts

**Phase 1 Statistics:**
- 18 papers processed (2017-2025)
- 11 unique datasets identified
- 5 robot platforms tracked
- 406 text chunks indexed
- < 0.1s semantic search latency

## 📊 Data Pipeline Summary

**Input:** Raw PDFs → Extract text → **Process:**

1. **Extract Structured Info** → CSV with 26 fields
   - Datasets, models, hardware, training details
   
2. **Populate Database** → SQLite with relational schema
   - Normalized tables, indexed for fast queries
   
3. **Build Vector Index** → FAISS for semantic search
   - 512-word chunks, 384-dim embeddings

**Output:** Queryable database + semantic search

## 🎯 Current Progress

- ✅ **Phase 1: Data Preparation** (Complete)
- 🔄 **Phase 2: Tool Implementation** (Next)
  - Database query tool
  - Vector search tool
  - PDF snippet extraction
  - Web search integration
- ⏳ **Phase 3: LLM Integration**
- ⏳ **Phase 4: Evaluation & Security**

## 💻 Technologies

- **Python 3.8+**
- **SQLite** - Relational database
- **FAISS** - Vector similarity search
- **Sentence Transformers** - Text embeddings
- **Regex** - Pattern-based extraction

## 📝 Development Notes

### Source Data
- `papers/` - Original PDFs (kept for reference)
- `processed_papers/` - Extracted text and metadata

### Phase 1 Package
- All Phase 1 code is in `phase1_data_preparation/`
- Self-contained with full documentation
- Ready for testing/evaluation

### Development Scripts
- `scripts/research_pdf_parser_clean.py` - For processing additional papers
- `scripts/query_database.py` - Database utilities
- `scripts/thesisaurus_va.ipynb` - Experimentation notebook

