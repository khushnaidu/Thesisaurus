import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from phase2_tools.query_planner import QueryPlanner
from phase2_tools.tool_registry import ToolRegistry
from phase3_llm.llm_wrapper import LLMWrapper
from phase3_llm.prompt_builder import PromptBuilder
from phase3_llm.security import InputGuard

class ResearchAssistant:
    def __init__(self, llm, db_path=None, index_path=None, metadata_path=None):
        self.llm = llm
        self.planner = QueryPlanner()
        self.registry = ToolRegistry(db_path, index_path, metadata_path)
        self.prompt_builder = PromptBuilder()
        self.guard = InputGuard()

        # toggles for the prompting techniques
        self.use_chaining = True
        self.use_reflection = True
        self.use_guard = True

        # just a quick lookup for tool types
        self.tool_types = {
            'get_all_datasets': 'Database',
            'get_all_vision_models': 'Database',
            'get_training_setups': 'Database',
            'get_all_hardware': 'Database',
            'get_papers_by_year': 'Database',
            'get_paper_metadata': 'Database',
            'search_papers_by_dataset': 'Database',
            'get_database_overview': 'Database',
            'semantic_search': 'RAG',
            'search_within_paper': 'RAG',
            'get_paper_chunks': 'RAG',
            'search_arxiv': 'Web',
            'get_arxiv_paper': 'Web',
            'search_by_author': 'Web',
            'search_recent_papers': 'Web',
        }

    def parse_llm_plan(self, resp):
        # pull out the tools from llm response
        tools = []
        reasoning = ""
        for line in resp.strip().split('\n'):
            if line.startswith('REASONING:'):
                reasoning = line.replace('REASONING:', '').strip()
            elif line.startswith('TOOLS:'):
                raw = line.replace('TOOLS:', '').strip()
                tools = [t.strip() for t in raw.split(',') if t.strip()]

        # fallback if it didnt parse right
        if not tools:
            tools = ['semantic_search']
        return {'tools': tools, 'reasoning': reasoning}

    def parse_reflection(self, resp):
        issues = ""
        verdict = "PASS"
        for line in resp.strip().split('\n'):
            if line.startswith('ISSUES:'):
                issues = line.replace('ISSUES:', '').strip()
            elif line.startswith('VERDICT:'):
                verdict = line.replace('VERDICT:', '').strip()
        return {'passed': 'PASS' in verdict.upper(), 'issues': issues}

    def answer(self, query):
        print(f"\n{'='*60}")
        print(f"query: {query}")
        print('='*60)

        # security check first
        if self.use_guard:
            safe, pattern = self.guard.check(query)
            if not safe:
                print(f"[guard] blocked query")
                return {
                    'query': query,
                    'answer': "cant process that for security reasons",
                    'blocked': True,
                    'pattern': pattern
                }

        # prompt chaining - let llm pick the tools
        if self.use_chaining:
            plan_prompt = self.prompt_builder.build_planning_prompt(query)
            plan_resp = self.llm.generate(plan_prompt, max_tokens=150)
            parsed = self.parse_llm_plan(plan_resp)
            print(f"\n[chain] reasoning: {parsed['reasoning']}")
            print(f"[chain] picked: {parsed['tools']}")

            plan = {'tools': [{'name': t, 'params': {'query': query}} for t in parsed['tools']]}
            tool_names = parsed['tools']
        else:
            plan = self.planner.plan(query)
            tool_names = [t['name'] for t in plan['tools']]

        ttypes = list(set([self.tool_types.get(n, n) for n in tool_names]))
        print(f"[tools] using: {ttypes}")

        # run the tools
        results = self.planner.execute_plan(plan, self.registry)
        num_success = sum(1 for r in results.values() if r.get('success'))
        print(f"[tools] got data from {num_success} tools")

        # build prompt and generate
        prompt = self.prompt_builder.build_prompt(query, results)
        print(f"[llm] generating...")
        answer = self.llm.generate(prompt, max_tokens=300)

        # self reflection - check if answer is good
        reflection = None
        if self.use_reflection:
            print(f"[reflect] checking answer...")
            ref_prompt = self.prompt_builder.build_reflection_prompt(query, answer, results)
            ref_resp = self.llm.generate(ref_prompt, max_tokens=150)
            reflection = self.parse_reflection(ref_resp)

            if reflection['passed']:
                print(f"[reflect] looks good")
            else:
                print(f"[reflect] found issues: {reflection['issues']}")
                # try again with stricter instructions
                strict = prompt + "\nonly use facts from sources. dont make stuff up."
                answer = self.llm.generate(strict, max_tokens=300)

        print(f"\n{'='*60}")
        print("answer:")
        print('='*60)
        print(answer)
        print('='*60 + '\n')

        return {
            'query': query,
            'answer': answer,
            'tools_used': tool_names,
            'tool_types': ttypes,
            'tool_results': results,
            'reflection': reflection
        }
