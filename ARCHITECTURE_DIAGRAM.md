# Architecture Diagrams for Presentation

## High-Level System Flow (Main Architecture Slide)

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER QUERY                              │
│              "What datasets are used for manipulation?"         │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      QUERY PLANNER                              │
│                                                                 │
│  • Analyze query intent (keywords, patterns)                   │
│  • Select appropriate tools                                    │
│  • Priority: Web Search → Semantic → Database                 │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                       TOOL REGISTRY                             │
│                                                                 │
│   ┌─────────────┐    ┌──────────────┐    ┌─────────────┐     │
│   │  Database   │    │ Vector Search│    │ Web Search  │     │
│   │   Tools     │    │   (RAG)      │    │   Tools     │     │
│   │             │    │              │    │             │     │
│   │ • datasets  │    │ • semantic   │    │ • arXiv     │     │
│   │ • models    │    │ • chunks     │    │ • recent    │     │
│   │ • training  │    │ • papers     │    │ • author    │     │
│   └──────┬──────┘    └──────┬───────┘    └──────┬──────┘     │
│          │                  │                   │             │
└──────────┼──────────────────┼───────────────────┼─────────────┘
           │                  │                   │
           ▼                  ▼                   ▼
    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
    │  papers.db  │    │FAISS Index  │    │ arXiv API   │
    │   (SQLite)  │    │+ metadata   │    │             │
    └─────────────┘    └─────────────┘    └─────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      TOOL RESULTS                               │
│                                                                 │
│  • Structured data from database                               │
│  • Semantic chunks from papers                                 │
│  • External papers from web                                    │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   PROMPT BUILDER                                │
│                                                                 │
│  • Format tool results for LLM                                 │
│  • Add system instructions                                     │
│  • Include user query context                                  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              LLAMA-3.1-8B-INSTRUCT (4-bit)                      │
│                                                                 │
│  • Synthesize coherent answer                                  │
│  • Cite relevant sources                                       │
│  • Avoid hallucination                                         │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    NATURAL LANGUAGE ANSWER                      │
│  "The most common datasets are BridgeData V2 and Open-X        │
│   Embodiment, both used in 5 papers..."                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3-Phase Pipeline (Simplified)

```
┌──────────────────────────────────────────────────────────────────┐
│                         PHASE 1                                  │
│                    DATA PREPARATION                              │
│                                                                  │
│  18 Papers  →  Manual Extraction  →  Storage                    │
│                                                                  │
│  • RT-1, RT-2, OpenVLA, Octo...                                 │
│  • Extract: datasets, models, training, hardware                │
│  • Store: SQLite DB + FAISS index + metadata                    │
└──────────────────────────────────────────────────────────────────┘
                             ↓
┌──────────────────────────────────────────────────────────────────┐
│                         PHASE 2                                  │
│                       TOOL LAYER                                 │
│                                                                  │
│  Query Planner  →  15 Specialized Tools  →  Results             │
│                                                                  │
│  • Keyword-based routing                                        │
│  • 8 database + 4 vector + 3 web tools                          │
│  • Execute and return structured results                        │
└──────────────────────────────────────────────────────────────────┘
                             ↓
┌──────────────────────────────────────────────────────────────────┐
│                         PHASE 3                                  │
│                    LLM INTEGRATION                               │
│                                                                  │
│  Tool Results  →  Prompt Building  →  LLM  →  Answer            │
│                                                                  │
│  • Format results for Llama-3.1-8B                              │
│  • Generate coherent natural language                           │
│  • Prevent hallucination with strict prompts                    │
└──────────────────────────────────────────────────────────────────┘
```

---

## Tool Categories (For Slide)

```
                    ┌─────────────────────────┐
                    │    TOOL REGISTRY        │
                    │    (15 Total Tools)     │
                    └───────────┬─────────────┘
                                │
                ┌───────────────┼───────────────┐
                │               │               │
                ▼               ▼               ▼
    ┌───────────────────┐ ┌─────────────┐ ┌─────────────┐
    │  DATABASE TOOLS   │ │ VECTOR SEARCH│ │ WEB SEARCH  │
    │       (8)         │ │     (4)      │ │     (3)     │
    ├───────────────────┤ ├─────────────┤ ├─────────────┤
    │• get_all_datasets │ │• semantic   │ │• search     │
    │• get_vision_models│ │  _search    │ │  _arxiv     │
    │• get_training     │ │• search     │ │• search     │
    │  _setups          │ │  _within    │ │  _recent    │
    │• get_hardware     │ │  _paper     │ │• search_by  │
    │• get_papers_by    │ │• get_paper  │ │  _author    │
    │  _year            │ │  _chunks    │ │             │
    │• get_paper        │ │• get_similar│ │             │
    │  _metadata        │ │  _papers    │ │             │
    │• search_papers_by │ │             │ │             │
    │  _dataset         │ │             │ │             │
    │• get_database     │ │             │ │             │
    │  _overview        │ │             │ │             │
    └───────────────────┘ └─────────────┘ └─────────────┘
```

---

## Query Flow Example (For Demo Slide)

```
Query: "What datasets are used for manipulation?"
   │
   ▼
[Query Planner]
   • Matches: 'datasets' keyword
   • Selects: Database Tool
   │
   ▼
[Tool: get_all_datasets]
   • Queries papers.db
   • Returns: BridgeData V2 (5 papers)
             Open-X Embodiment (5 papers)
             CALVIN (3 papers)...
   │
   ▼
[Prompt Builder]
   • Formats results
   • Adds system prompt
   • Includes user query
   │
   ▼
[Llama-3.1-8B]
   • Synthesizes answer
   • Natural language output
   │
   ▼
"The most common evaluation datasets are BridgeData V2 
and Open-X Embodiment, both used in 5 papers..."
```

---

## Data Storage Architecture

```
┌────────────────────────────────────────────────────────────┐
│                    18 ROBOTICS PAPERS                      │
│   RT-1, RT-2, OpenVLA, Octo, GR00T, RoboGen, Rialto...    │
└───────────────────────┬────────────────────────────────────┘
                        │
            ┌───────────┴───────────┐
            │                       │
            ▼                       ▼
┌───────────────────────┐  ┌──────────────────────┐
│  STRUCTURED DATA      │  │    TEXT CONTENT      │
│                       │  │                      │
│  • Datasets           │  │  • Full paper text   │
│  • Vision models      │  │  • Chunked (512 tok) │
│  • Training configs   │  │  • Embedded (384-dim)│
│  • Hardware/robots    │  │                      │
│  • Year, authors...   │  │                      │
└───────┬───────────────┘  └──────┬───────────────┘
        │                         │
        ▼                         ▼
┌───────────────────┐    ┌───────────────────┐
│   papers.db       │    │   FAISS Index     │
│   (SQLite)        │    │   + metadata.json │
│                   │    │                   │
│  • Fast queries   │    │  • Semantic search│
│  • Structured     │    │  • Similarity     │
└───────────────────┘    └───────────────────┘
```

---

## Color Coding Suggestions

- **Database Tools**: Blue (#4285F4)
- **Vector Search**: Green (#34A853)
- **Web Search**: Orange (#FBBC04)
- **LLM**: Purple (#A142F4)
- **Data Storage**: Gray (#5F6368)

---

## Key Metrics for Slides

```
┌──────────────────────────────────────────┐
│          SYSTEM STATISTICS               │
├──────────────────────────────────────────┤
│  📚 Papers Processed:         18         │
│  🔧 Total Tools:              15         │
│  🗄️  Database Tools:          8          │
│  🔍 Vector Search Tools:      4          │
│  🌐 Web Search Tools:         3          │
│  🤖 LLM Model:          Llama-3.1-8B     │
│  ⚡ Quantization:        4-bit (NF4)     │
│  💾 Model Size:         ~4.5 GB          │
│  ⏱️  Query Latency:     3-5 seconds      │
│  🎯 Query Success:      100%             │
└──────────────────────────────────────────┘
```

---

## PowerPoint/Slides Tips

### Slide 4: System Architecture
- Use the "High-Level System Flow" diagram
- Animate each component appearing sequentially
- Use arrows to show data flow

### Slide 5: 3-Phase Pipeline
- Use the "3-Phase Pipeline" diagram
- Keep it simple and high-level
- Add phase numbers: 1️⃣ 2️⃣ 3️⃣

### Slide 6: Tool Categories
- Use the "Tool Categories" tree diagram
- Color-code each category
- Show number of tools in each

### Slide 7: Live Demo
- Use "Query Flow Example" diagram
- Run actual Colab demo alongside
- Show input → process → output

---

Good luck! 🚀

