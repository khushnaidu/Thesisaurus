# phase 1: data preparation

data prep pipeline for the rag system. extracts structured info from papers, builds a database, and creates a vector index for semantic search.

## structure

```
phase1_data_preparation/
├── data/
│   ├── full_text/           # paper text files
│   └── metadata.json        # paper metadata
├── scripts/
│   ├── 1_extract_structured_info.py
│   ├── 2_populate_database.py
│   └── 3_build_vector_index.py
└── outputs/
    ├── extracted_info.csv
    ├── papers.db
    ├── faiss_index.bin
    └── chunk_metadata.json
```

## setup

```bash
pip install -r requirements.txt
```

deps: sentence-transformers, faiss-cpu, numpy, torch, transformers

## running the pipeline

```bash
cd scripts/

# step 1: extract structured info
python 1_extract_structured_info.py

# step 2: populate database
python 2_populate_database.py

# step 3: build vector index
python 3_build_vector_index.py
```

## what each step does

### step 1: extraction
extracts from paper text using regex:
- datasets (bridgedata, droid, open-x embodiment, etc)
- models (rt-1, openvla, octo)
- hardware (franka panda, a100, etc)
- training params (optimizer, lr, batch size)

### step 2: database
creates sqlite db with normalized schema:
- papers table (main info)
- datasets, robots, hardware tables
- junction tables for many-to-many

### step 3: vector index
builds faiss index for semantic search:
- chunks papers into 512 word segments
- embeds with all-MiniLM-L6-v2
- creates flat L2 index

## example queries

database:
```sql
SELECT p.title FROM papers p
JOIN paper_datasets pd ON p.paper_id = pd.paper_id
JOIN datasets d ON pd.dataset_id = d.id
WHERE d.name = 'BridgeData V2';
```

semantic search:
```python
builder.search("vision-language-action models", top_k=3)
```

## stats

- 18 papers processed
- ~406 chunks indexed
- 11 unique datasets
- search latency < 0.1s
