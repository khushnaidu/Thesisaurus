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

The whole point of this phase is to turn a pile of PDFs into something an LLM can actually work with.

**Text Extraction**: I used PyMuPDF instead of other PDF libraries because it handles the messy formatting of academic papers way better. Most papers have two-column layouts, weird headers/footers, and embedded figures that break simpler parsers. PyMuPDF gives cleaner raw text.

**Section Parsing**: Rather than treating each paper as one giant blob of text, I parse it into sections (abstract, introduction, methods, experiments, results, etc.). This matters because when someone asks about "experimental results," I don't want to retrieve chunks from the abstract. Section awareness makes retrieval way more relevant.

**Chunking Strategy**: The extracted text gets split into chunks for the vector index. I went with ~500 token chunks with overlap, and each chunk keeps metadata about which paper and section it came from. This way when we retrieve context for the LLM, we know exactly where it's from and can cite it properly.

**Database Population**: Some info is better stored in a structured database than a vector index. Things like "which papers use CLIP" or "list all datasets" are exact lookups, not semantic searches. So I extract entities (datasets, robots, vision models, hardware) using pattern matching and store them in SQLite with proper relations. This lets us do fast exact queries without burning tokens on retrieval.

**Vector Index**: For everything else, FAISS handles semantic search. I use `all-MiniLM-L6-v2` from sentence-transformers to embed chunks. When a user asks a question, we embed their query and find the most similar chunks. The section metadata helps us return results like `[openvla (experiments)]` instead of just `[openvla]`.

### Phase 2: Tools

The LLM doesn't query data directly - it calls tools. This is cleaner and lets us control exactly what the model can access.

**Database Tool**: Wraps SQLite queries. Has functions like `get_all_datasets()`, `get_all_vision_models()`, `get_papers_by_year()`, etc. When someone asks "what datasets are used," we don't need semantic search - just query the DB directly. Way faster and more accurate for structured lookups.

**Vector Search Tool**: This is the RAG component. Takes a query, embeds it, finds top-k similar chunks from the FAISS index, and returns them with metadata. Also has `search_within_paper()` for when you want to search only one specific paper.

**Web Tool**: Wraps the arXiv API. Sometimes users ask about papers outside our corpus or want recent work on a topic. This lets the assistant search arXiv, fetch paper metadata, and include external sources in its answers.

**Tool Registry**: All tools register in one place so the query planner can pick from them. Each tool has a name and description that gets shown to the LLM during planning.

### Phase 3: LLM Integration

This is where it all comes together. The `ResearchAssistant` class orchestrates everything.

**Query Planning**: When a question comes in, the LLM first decides which tools to use. I give it a list of available tools with descriptions, and it outputs something like `TOOLS: get_all_datasets, semantic_search`. There's also a keyword-based fallback in case the LLM misses obvious ones (like if the query mentions "datasets" but the LLM only picks semantic_search).

**Prompt Chaining**: For complex questions, we break them down. The LLM decomposes "What datasets are popular and which papers use CLIP?" into two sub-questions, answers each one separately (with its own tool calls), then synthesizes a final answer. This works better than trying to answer everything at once.

**Meta-Prompting**: Different questions need different response styles. A comparison question should list similarities and differences. A "list all X" question should use bullet points. I detect the query type and inject the appropriate policy into the prompt so the LLM formats its response correctly.

**Self-Reflection**: After generating an answer, the LLM checks its own work. It looks at the source data and asks: does this answer the question? Are the claims supported? Any hallucinations? If it finds issues, we regenerate with stricter instructions. This catches a lot of mistakes.

**Prompt Caching**: LLM calls are slow and cost money. If we've seen the exact same prompt before, we just return the cached response. Simple hash-based cache, but it makes repeated queries instant.

**Security (InputGuard)**: Users can be creative with prompts. The guard checks for injection attempts - things like "ignore previous instructions" or SQL injection patterns. Honestly, for an academic project the threat model is limited, but it's good practice and demonstrates awareness of prompt injection risks. The philosophy is simple: whitelist expected patterns, reject anything that looks like it's trying to break out of the assistant role.

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
