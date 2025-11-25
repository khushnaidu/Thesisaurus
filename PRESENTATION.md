# Thesaurus: RAG Research Assistant for Robotics Papers
## 5-Minute Presentation Outline

---

## SLIDE 1: Title Slide (5 seconds)
**Thesaurus: AI Research Assistant for Robotics Papers**
- Your Name
- NLP Project - Fall 2025

---

## SLIDE 2-3: Motivation (1 minute)

### The Problem
**"Robotics researchers are drowning in papers"**
- 1000+ new robotics papers published monthly on arXiv alone
- Finding relevant datasets, models, and techniques takes hours
- Information scattered across PDFs, tables, and supplementary materials

### The Solution
**An AI assistant that understands and answers questions about robotics research**
- "What datasets are used for manipulation?"
- "Tell me about sim-to-real transfer"
- "Find recent papers on vision-language models"

### Why This Matters
- Accelerates literature review from hours → seconds
- Makes research more accessible to students and newcomers
- Connects structured data with semantic understanding

---

## SLIDE 4-6: Approach & Architecture (2 minutes)

### System Overview
**3-Phase RAG Pipeline**

```
User Query → Query Planner → Tool Selection → LLM Synthesis → Natural Language Answer
```

### Phase 1: Data Preparation
**18 Robotics Papers (2017-2025)**
- Manual extraction: datasets, models, training configs, hardware
- Stored in SQLite database (structured queries)
- Text chunked & embedded with FAISS index (semantic search)
- ArXiv API integration (external search)

**Key Papers:** RT-1, RT-2, OpenVLA, Octo, GR00T, RoboGen...

### Phase 2: Tool Layer (15 Specialized Tools)
**Three Tool Categories:**
1. **Database Tools (8)** → Structured queries
   - get_all_datasets, get_all_vision_models, get_training_setups...
2. **Vector Search Tools (4)** → Semantic search
   - semantic_search, search_within_paper...
3. **Web Search Tools (3)** → External sources
   - search_arxiv, search_recent_papers...

**Smart Query Routing:**
- Keyword pattern matching
- Priority: Web > Semantic > Database
- Automatic tool selection based on query intent

### Phase 3: LLM Integration
**Llama-3.1-8B-Instruct (4-bit quantization)**
- Query Planning: Routes queries to appropriate tools
- Result Synthesis: Combines tool outputs into coherent answers
- Prompt Engineering: Prevents hallucination, ensures accuracy

**Technical Highlights:**
- 4-bit quantization (bitsandbytes) → Runs on T4 GPU
- Custom prompt templates for each tool type
- Module caching for fast iteration in Colab

---

## SLIDE 7: Live Demo (30 seconds)

**Show 3 Query Types:**

1. **Database Query:**
   - "What are the most common evaluation datasets?"
   - Shows: BridgeData V2, Open-X Embodiment...

2. **Semantic Search:**
   - "Tell me about OpenVLA"
   - Pulls from paper content, explains the model

3. **Web Search:**
   - "Look up recent papers on sim-to-real learning"
   - Fetches from arXiv API with titles + links

**Output Format:**
- 🔍 Query → 📋 Tools Selected → ✓ Data Retrieved → 💡 Answer
- Clean, easy-to-read presentation format

---

## SLIDE 8: Next Steps (1 minute)

### Immediate Improvements
1. **Expand Dataset**
   - 18 → 100+ papers
   - More domains: manipulation, navigation, learning

2. **Advanced Query Planning**
   - LLM-based routing (replace keyword matching)
   - Multi-step reasoning for complex queries

3. **Enhanced RAG**
   - Reranking with cross-encoders
   - Query expansion for better retrieval

### Future Enhancements
4. **Model Benchmarking**
   - Compare Llama-3.1-8B vs Llama-3.3-70B
   - Evaluate answer quality (RAGAS, human eval)

5. **Production Features**
   - Caching for repeated queries
   - Citation tracking (which paper said what)
   - Interactive web interface

6. **Research Applications**
   - Paper recommendation system
   - Automatic related work generation
   - Methodology comparison across papers

---

## SLIDE 9: Conclusion (10 seconds)

**Thesaurus: Making Robotics Research Accessible**
- 🤖 3-phase RAG pipeline
- 🔧 15 specialized tools
- 📚 Structured + Semantic + Web search
- ⚡ Fast, accurate, and demo-ready

**GitHub:** github.com/khushnaidu/Thesisaurus

---

## Q&A Preparation (1 minute)

### Expected Questions & Answers

**Q: Why not just use ChatGPT with RAG?**
A: ChatGPT doesn't have access to specialized robotics databases or structured metadata (datasets, training configs). Our system combines structured queries with semantic search for more precise answers.

**Q: How do you prevent hallucination?**
A: 3 ways:
1. Strict prompt engineering: "Only use provided information"
2. Tool-based answers: LLM synthesizes from tool results, not memory
3. 4-bit quantization maintains model quality while reducing memory

**Q: Why Llama instead of OpenAI?**
A: 
1. Local deployment (no API costs)
2. Reproducibility for research
3. Planned benchmarking (8B vs 70B)
4. Open-source transparency

**Q: How accurate is the query routing?**
A: Currently keyword-based (~80% accurate). Next step: LLM-based routing for better intent understanding.

**Q: Can it handle multi-step reasoning?**
A: Not yet - that's a key next step. Currently handles single-shot queries well.

**Q: How did you extract structured data?**
A: Manual extraction from papers (datasets, models, configs) + automated chunking for semantic search. Future: automated extraction with NER/table parsing.

**Q: What's the latency?**
A: ~3-5 seconds per query (tool execution + LLM inference on T4). Caching would reduce this.

**Q: Why these 18 papers?**
A: Representative sample of recent robotics learning (2017-2025): manipulation, VLMs, sim-to-real, foundation models. Easy to expand.

---

## Presentation Tips

### Timing Breakdown
- Slide 1: 5 sec (Title)
- Slides 2-3: 1 min (Motivation)
- Slides 4-6: 2 min (Approach)
- Slide 7: 30 sec (Demo)
- Slide 8: 1 min (Next Steps)
- Slide 9: 10 sec (Conclusion)
- **Total: 4 min 45 sec** → Leaves 15 sec buffer

### Delivery Notes
- **Start strong:** "Robotics researchers publish 1000+ papers monthly - finding relevant info is overwhelming"
- **Show, don't tell:** Run live demo in Colab for impact
- **Be confident:** This is a working system, not just a proposal
- **End with impact:** "From hours of literature review to seconds of intelligent search"

### Slide Design Tips
- Use diagrams for architecture (boxes + arrows)
- Show example queries + outputs
- Include screenshots from Colab demo
- Keep text minimal, speak the details
- Use emojis sparingly for visual interest

---

## Quick Reference: Key Numbers

- **18** papers processed
- **15** specialized tools (8 database, 4 vector, 3 web)
- **3** tool categories
- **8B** parameters in Llama model (4-bit quantized)
- **3-5 sec** query latency
- **100%** demo success rate (after fixes!)

---

Good luck with your presentation! 🚀

