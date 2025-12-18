# Thesisaurus

**CMPE 259 - NLP Final Project**
**Khush Naidu** | San Jose State University

A research assistant for querying and synthesizing information from academic papers.

### [View the Demo Notebook on Colab](https://colab.research.google.com/drive/1Um-YM0SjsKcdULEzJx_rQQ1F9VdiGS9D)

The idea is simple: researchers spend too much time manually organizing papers, comparing experimental results, and tracking details across a growing corpus. This tool uses RAG + LLM to answer questions about a collection of robotics/embodied AI papers.

## What it does

- Answers questions about papers using a combination of database queries, semantic search, and web lookups
- Compares methods, datasets, and results across papers
- Retrieves specific details (training setups, hardware, benchmarks)
- Searches arXiv for related external work

## Project Structure

```
Thesisaurus/
├── phase1_data_preparation/    # pdf extraction, database, vector index
│   ├── scripts/                # extraction and indexing scripts
│   ├── data/                   # extracted text and sections
│   └── outputs/                # papers.db, faiss_index.bin
├── phase2_tools/               # tool implementations
│   ├── database_tool.py        # sqlite queries
│   ├── vector_search_tool.py   # faiss semantic search
│   └── web_search_tool.py      # arxiv api
├── phase3_llm/                 # llm pipeline
│   ├── llm_wrapper.py          # model loading (local + api)
│   ├── pipeline.py             # main ResearchAssistant class
│   ├── prompt_builder.py       # prompt templates
│   └── security.py             # input guard
├── papers/                     # source pdfs (18 papers)
└── Thesisaurus_NLP_Project.ipynb  # main demo notebook
```

## Pipeline

### Phase 1: Data Preparation
- Extract text from PDFs using PyMuPDF
- Parse sections (abstract, methods, experiments, etc.)
- Populate SQLite database with structured info (datasets, robots, hardware, vision models)
- Build FAISS vector index with section-aware chunks (557 chunks)

### Phase 2: Tools
Three tool types that the LLM can call:
- **Database**: queries for datasets, vision models, robots, hardware, papers by year
- **RAG**: semantic search over paper chunks using sentence-transformers
- **Web**: arXiv API for external paper lookups

### Phase 3: LLM Integration
- Query planning: LLM decides which tools to use
- Prompt chaining: decompose complex queries into sub-questions
- Meta-prompting: response policies based on query type (comparison, list, explain)
- Self-reflection: verify answers against source data
- Prompt caching: avoid redundant API calls

## Models

- **Small**: Llama-3.1-8B-Instruct (local, 4-bit quantized)
- **Large**: Llama-3.3-70B-Instruct (via Together API)

## Stats

- 18 papers (2020-2024)
- 557 indexed chunks
- 13 datasets, 11 robots, 7 vision models tracked
- ~3s response time (70B), ~0.001s cached

## Running it

The main demo is in `Thesisaurus_NLP_Project.ipynb` (designed for Google Colab with GPU).

To run locally:
```bash
pip install -r phase1_data_preparation/requirements.txt
pip install transformers torch faiss-cpu sentence-transformers

# rebuild index if needed
cd phase1_data_preparation/scripts
python extract_with_pymupdf.py
python 2_populate_database.py
python build_section_aware_index.py
```

## Tech

Python, SQLite, FAISS, sentence-transformers, Hugging Face transformers, Together API, arXiv API
