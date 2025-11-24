# NLP Project: Robotics Paper Analysis

A project for extracting and analyzing information from robotics research papers.

## Project Structure

```
NLP_Project/
├── papers/                    # Raw PDF files
├── processed_papers/          # Extracted text and metadata
│   ├── full_text/            # Full text from PDFs
│   └── metadata.json         # Paper metadata
├── extract_structured_info.py # Phase 1.2: Extract structured data
├── extracted_info.csv        # Structured data in CSV format
├── populate_database.py      # Phase 1.3: Create and populate database
├── query_database.py         # Helper script for querying database
└── papers.db                 # SQLite database
```

## Phase 1: Data Collection & Processing

### Phase 1.1: PDF Processing ✓
- Extract text and metadata from research papers
- Store in `processed_papers/` directory

### Phase 1.2: Structured Information Extraction ✓
Extract key information from papers:
- **Datasets**: Training and evaluation datasets used
- **Models**: Model names, architectures, sizes, vision encoders
- **Hardware**: Robot platforms, sensors, grippers, GPUs/TPUs
- **Training**: Optimizers, learning rates, batch sizes, epochs
- **Evaluation**: Tasks, success rates, baseline comparisons

**Usage:**
```bash
python extract_structured_info.py processed_papers extracted_info.csv
```

**Output**: `extracted_info.csv` with 27 fields per paper

### Phase 1.3: Database Population ✓
Store structured data in a SQLite database for easy querying.

**Database Schema:**
- `papers` - Main paper information
- `datasets` - Unique datasets (many-to-many with papers)
- `robots` - Robot platforms (many-to-many with papers)
- `hardware` - Sensors, grippers, GPUs (many-to-many with papers)
- `baselines` - Baseline models for comparison
- `tasks` - Evaluation tasks per paper

**Usage:**
```bash
# Create and populate database
python populate_database.py extracted_info.csv papers.db

# Query database with helper script
python query_database.py

# Or use sqlite3 directly
sqlite3 papers.db "SELECT * FROM papers LIMIT 5;"
```

## Current Dataset Stats

- **Papers**: 18 papers (2017-2025)
- **Datasets**: 11 unique datasets
  - Most popular: Open-X Embodiment (5 papers), BridgeData V2 (5 papers)
- **Robot Platforms**: 5 robots
  - Most common: Franka Panda (4 papers), Aloha (4 papers)
- **Models with sizes**: 1 (OpenVLA: 7B parameters)
- **Success rates reported**: 1 paper (OKAMI: 84.0%)

## Example Queries

### Most popular datasets:
```sql
SELECT d.name, COUNT(*) as count 
FROM datasets d
JOIN paper_datasets pd ON d.id = pd.dataset_id
GROUP BY d.name
ORDER BY count DESC;
```

### Papers using specific robot:
```sql
SELECT p.paper_id, p.title, p.year
FROM papers p
JOIN paper_robots pr ON p.paper_id = pr.paper_id
JOIN robots r ON pr.robot_id = r.id
WHERE r.name = 'Franka Panda';
```

### Models by size:
```sql
SELECT paper_id, model_name, model_size, year
FROM papers
WHERE model_size IS NOT NULL
ORDER BY year DESC;
```

## Technologies Used

- **Python 3.x**
- **SQLite** - Lightweight database for structured storage
- **CSV** - Intermediate data format
- **Regex** - Pattern matching for information extraction

## Next Steps

- Phase 2: Statistical Analysis & Visualizations
- Phase 3: Trend Analysis over time
- Phase 4: Citation network analysis (if needed)

## Notes

- Some papers have incomplete metadata due to PDF parsing limitations
- Success rates and model sizes are only available when explicitly mentioned in papers
- Dataset associations include both training and evaluation datasets

