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
        # pull out tools, enforce max 4, no duplicates
        tools = []
        for line in resp.strip().split('\n'):
            if line.upper().startswith('TOOLS:'):
                raw = line.split(':', 1)[1].strip()
                for chunk in raw.split(','):
                    chunk = chunk.strip().lower()
                    for known in self.tool_types.keys():
                        if known in chunk and known not in tools:
                            tools.append(known)
                            break
                    if len(tools) >= 4:
                        break
                break

        if not tools:
            tools = ['semantic_search']
        return {'tools': tools[:4]}

    def parse_reflection(self, resp):
        issues = ""
        verdict = "PASS"
        for line in resp.strip().split('\n'):
            if line.startswith('ISSUES:'):
                issues = line.replace('ISSUES:', '').strip()
            elif line.startswith('VERDICT:'):
                verdict = line.replace('VERDICT:', '').strip()
        return {'passed': 'PASS' in verdict.upper(), 'issues': issues}

    def parse_decomposition(self, resp):
        # pull out sub-questions, max 2, must be actual questions
        steps = []
        for line in resp.strip().split('\n'):
            if line.upper().startswith('STEP1:') or line.upper().startswith('STEP2:'):
                q = line.split(':', 1)[1].strip()
                # must be a question (ends with ?) and not "none"
                if q and q.endswith('?') and 'none' not in q.lower():
                    steps.append(q)
                if len(steps) >= 2:
                    break
        return steps

    def answer_with_chaining(self, query):
        # full chain: break down query -> answer each part -> combine
        print(f"\n{'='*60}")
        print(f"prompt chaining: {query}")
        print('='*60)

        if self.use_guard:
            safe, _ = self.guard.check(query)
            if not safe:
                return {'query': query, 'answer': "blocked", 'blocked': True}

        # break the query into smaller questions
        print("\n[step 1] breaking down query...")
        decomp_prompt = self.prompt_builder.build_decomposition_prompt(query)
        decomp_resp = self.llm.generate(decomp_prompt, max_tokens=200)
        sub_qs = self.parse_decomposition(decomp_resp)

        if not sub_qs:
            sub_qs = [query]

        print(f"got {len(sub_qs)} sub-questions:")
        for i, sq in enumerate(sub_qs, 1):
            print(f"  {i}. {sq}")

        # answer each one, using previous answers as context
        answers = []
        tools_used = []

        for i, sq in enumerate(sub_qs, 1):
            print(f"\n[step {i+1}] answering: {sq[:50]}...")

            # figure out what tools to use
            plan_prompt = self.prompt_builder.build_planning_prompt(sq)
            plan_resp = self.llm.generate(plan_prompt, max_tokens=150)
            parsed = self.parse_llm_plan(plan_resp)
            print(f"  tools: {parsed['tools']}")

            # run em
            plan = {'tools': [{'name': t, 'params': {'query': sq}} for t in parsed['tools']]}
            results = self.planner.execute_plan(plan, self.registry)
            tools_used.extend(parsed['tools'])

            # get answer with context from previous steps
            step_prompt = self.prompt_builder.build_substep_prompt(sq, answers, results)
            ans = self.llm.generate(step_prompt, max_tokens=200)
            answers.append(ans)
            print(f"  got: {ans[:80]}...")

        # put it all together
        print(f"\n[final] combining answers...")
        synth_prompt = self.prompt_builder.build_synthesis_prompt(query, answers)
        final = self.llm.generate(synth_prompt, max_tokens=300)

        print(f"\n{'='*60}")
        print("final answer:")
        print('='*60)
        print(final)
        print('='*60 + '\n')

        return {
            'query': query,
            'sub_questions': sub_qs,
            'step_answers': answers,
            'answer': final,
            'tools_used': list(set(tools_used)),
            'chain_length': len(sub_qs) + 2
        }

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

        # let llm pick the tools
        if self.use_chaining:
            plan_prompt = self.prompt_builder.build_planning_prompt(query)
            plan_resp = self.llm.generate(plan_prompt, max_tokens=100)
            parsed = self.parse_llm_plan(plan_resp)
            print(f"\n[tools] picked: {parsed['tools']}")

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
