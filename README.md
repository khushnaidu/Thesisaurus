# thesaurus: research assistant for robotics papers

rag system for querying and analyzing robotics research papers. built for cmpe 259.

## project structure

```
NLP_Project/
├── phase1_data_preparation/   # data pipeline (extract, db, index)
├── phase2_tools/              # tool implementations (db, vector, web)
├── phase3_llm/                # llm integration + prompting
├── papers/                    # source pdfs
├── processed_papers/          # extracted text
├── scripts/                   # utilities
└── venv/                      # virtual env
```

## quick start

```bash
# install deps
pip install -r requirements.txt

# run the pipeline
cd phase1_data_preparation/scripts
python 1_extract_structured_info.py
python 2_populate_database.py
python 3_build_vector_index.py
```

## phases

### phase 1: data prep
- text extraction from 18 papers
- structured info extraction (26 fields)
- sqlite database (11 tables)
- faiss vector index (406 chunks)

### phase 2: tools
- database queries
- semantic search (rag)
- arxiv api integration

### phase 3: llm
- llama 3.1 8b integration
- prompt chaining
- meta-prompting
- self-reflection

## stats

- 18 papers (2017-2025)
- 11 datasets identified
- 5 robot platforms
- <0.1s search latency

## tech stack

python, sqlite, faiss, sentence-transformers, huggingface transformers
