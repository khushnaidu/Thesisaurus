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
    
    def answer(self, query):
        """Main method: takes query, returns answer"""
        print(f"\n{'='*60}")
        print(f"QUERY: {query}")
        print('='*60)
        
        # Step 1: Plan which tools to use
        print("\n[1/4] Planning...")
        plan = self.planner.plan(query)
        tool_names = [t['name'] for t in plan['tools']]
        print(f"Will use: {tool_names}")
        
        # Step 2: Execute tools
        print("\n[2/4] Running tools...")
        tool_results = self.planner.execute_plan(plan, self.registry)
        success_count = sum(1 for r in tool_results.values() if r.get('success'))
        print(f"Success: {success_count}/{len(tool_results)} tools")
        
        # Step 3: Build prompt
        print("\n[3/4] Building prompt...")
        prompt = self.prompt_builder.build_prompt(query, tool_results)
        print(f"Prompt length: {len(prompt)} chars")
        
        # Step 4: Generate answer
        print("\n[4/4] Generating answer...")
        answer = self.llm.generate(prompt, max_tokens=300)
        
        print(f"\n{'='*60}")
        print("ANSWER:")
        print('='*60)
        print(answer)
        print('='*60)
        
        return {
            'query': query,
            'answer': answer,
            'tools_used': tool_names,
            'tool_results': tool_results
        }

