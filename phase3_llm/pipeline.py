import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from phase2_tools.query_planner import QueryPlanner
from phase2_tools.tool_registry import ToolRegistry
from phase3_llm.llm_wrapper import LLMWrapper
from phase3_llm.prompt_builder import PromptBuilder

class ResearchAssistant:
    def __init__(self, llm, db_path=None, index_path=None, metadata_path=None):
        """
        llm: LLMWrapper instance (already loaded)
        db_path: path to papers.db
        index_path: path to faiss_index.bin
        metadata_path: path to chunk_metadata.json
        """
        self.llm = llm
        self.planner = QueryPlanner()
        self.registry = ToolRegistry(db_path, index_path, metadata_path)
        self.prompt_builder = PromptBuilder()
        
        self.tool_types = {
            'get_all_datasets': 'Database Tool',
            'get_all_vision_models': 'Database Tool',
            'get_training_setups': 'Database Tool',
            'get_all_hardware': 'Database Tool',
            'get_papers_by_year': 'Database Tool',
            'get_paper_metadata': 'Database Tool',
            'search_papers_by_dataset': 'Database Tool',
            'get_database_overview': 'Database Tool',
            'semantic_search': 'Vector Search (RAG)',
            'search_within_paper': 'Vector Search (RAG)',
            'get_paper_chunks': 'Vector Search (RAG)',
            'search_arxiv': 'Web Search (arXiv)',
            'get_arxiv_paper': 'Web Search (arXiv)',
            'search_by_author': 'Web Search (arXiv)',
            'search_recent_papers': 'Web Search (arXiv)',
        }
    
    def answer(self, query):
        """Main method: takes query, returns answer"""
        print(f"\n{'='*70}")
        print(f"🔍 Query: {query}")
        print('='*70)
        
        # Step 1: Plan which tools to use
        plan = self.planner.plan(query)
        tool_names = [t['name'] for t in plan['tools']]
        tool_types = list(set([self.tool_types.get(name, name) for name in tool_names]))
        
        print(f"\n📋 Tools Selected: {', '.join(tool_types)}")
        
        # Step 2: Execute tools
        tool_results = self.planner.execute_plan(plan, self.registry)
        success_count = sum(1 for r in tool_results.values() if r.get('success'))
        print(f"✓ Retrieved data from {success_count} tool(s)")
        
        # Step 3: Build prompt
        prompt = self.prompt_builder.build_prompt(query, tool_results)
        
        # Step 4: Generate answer
        print(f"🤖 Generating answer with LLM...")
        answer = self.llm.generate(prompt, max_tokens=300)
        
        print(f"\n{'='*70}")
        print("💡 Answer:")
        print('='*70)
        print(answer)
        print('='*70 + '\n')
        
        return {
            'query': query,
            'answer': answer,
            'tools_used': tool_names,
            'tool_types': tool_types,
            'tool_results': tool_results
        }

