class PromptBuilder:
    def __init__(self):
        self.sys_prompt = """You are a research assistant that answers questions about robotics papers.
Answer the question directly using the tool results provided.
Keep your answer concise and factual.
Do not ask follow-up questions or generate additional Q&A examples."""

        # tools the llm can pick from
        self.tool_catalog = [
            {"name": "get_all_datasets", "desc": "list all datasets with usage counts"},
            {"name": "get_all_vision_models", "desc": "list vision models used in papers"},
            {"name": "get_training_setups", "desc": "get optimizer, lr, batch size etc"},
            {"name": "get_all_hardware", "desc": "list robots and hardware"},
            {"name": "semantic_search", "desc": "search paper content by meaning"},
            {"name": "search_arxiv", "desc": "search arxiv for papers"},
        ]

    def get_meta_policy(self, query):
        # meta prompting - different policies for different query types
        q = query.lower()

        if any(w in q for w in ['compare', 'vs', 'difference', 'versus']):
            return "comparison mode: present side by side, highlight differences"
        elif any(w in q for w in ['list', 'all', 'what are', 'which']):
            return "list mode: be thorough, use bullet points"
        elif any(w in q for w in ['how', 'why', 'explain', 'what is']):
            return "explanation mode: explain clearly, cite papers"
        elif any(w in q for w in ['summarize', 'overview', 'brief']):
            return "summary mode: keep it short, main points only"
        else:
            return "default: answer directly from sources"

    def build_planning_prompt(self, query):
        # prompt chaining step 1 - let llm decide tools
        tools = "\n".join([f"- {t['name']}: {t['desc']}" for t in self.tool_catalog])
        return f"""pick tools to answer this query.

available tools:
{tools}

query: {query}

respond in EXACTLY this format (no extra text):
REASONING: one short sentence
TOOLS: tool_name1, tool_name2

example response:
REASONING: user wants dataset info
TOOLS: get_all_datasets

your response:"""

    def format_tool_results(self, results):
        out = ""
        for name, res in results.items():
            if not res.get('success'):
                continue
            out += f"\n--- {name} ---\n"

            if 'datasets' in res:
                out += self._fmt_datasets(res['datasets'])
            elif 'vision_models' in res:
                out += self._fmt_models(res['vision_models'])
            elif 'training_setups' in res:
                out += self._fmt_training(res['training_setups'])
            elif 'hardware' in res:
                out += self._fmt_hardware(res['hardware'])
            elif 'results' in res:
                out += self._fmt_search(res['results'])
            elif 'paper' in res:
                out += self._fmt_paper(res['paper'])
            else:
                out += str(res)
        return out

    def _fmt_datasets(self, datasets):
        txt = "Datasets:\n"
        for d in datasets[:10]:
            name = d.get('name') or d.get('dataset_name', '?')
            cnt = d.get('paper_count', 0)
            txt += f"- {name}: {cnt} papers\n"
        return txt

    def _fmt_models(self, models):
        txt = "Vision models:\n"
        for m in models[:10]:
            name = m.get('model_name') or m.get('vision_encoder', '?')
            cnt = m.get('paper_count', 0)
            txt += f"- {name}: {cnt} papers\n"
        return txt

    def _fmt_training(self, setups):
        txt = "Training configs:\n"
        for s in setups[:5]:
            txt += f"\n{s.get('paper_id')}:\n"
            if s.get('optimizer'): txt += f"  opt: {s['optimizer']}\n"
            if s.get('learning_rate'): txt += f"  lr: {s['learning_rate']}\n"
            if s.get('batch_size'): txt += f"  batch: {s['batch_size']}\n"
        return txt

    def _fmt_hardware(self, hw):
        txt = "Hardware:\n"
        for h in hw[:10]:
            name = h.get('hardware_name') or h.get('robot_platform', '?')
            cnt = h.get('paper_count', 0)
            txt += f"- {name}: {cnt} papers\n"
        return txt

    def _fmt_search(self, results):
        txt = "Search results:\n"
        for i, r in enumerate(results[:5], 1):
            paper = r.get('paper_id', '?')
            content = r.get('text', r.get('content', ''))[:300]
            txt += f"\n{i}. [{paper}]: {content}...\n"
        return txt

    def _fmt_paper(self, p):
        txt = f"Paper: {p.get('title', p.get('paper_id'))}\n"
        if 'year' in p: txt += f"Year: {p['year']}\n"
        if 'abstract' in p: txt += f"Abstract: {p['abstract'][:300]}...\n"
        return txt

    def build_prompt(self, query, tool_results):
        formatted = self.format_tool_results(tool_results)
        policy = self.get_meta_policy(query)

        return f"""{self.sys_prompt}

policy: {policy}

question: {query}

tool results:
{formatted}

answer using only the info above. follow the policy.

answer:"""

    def build_reflection_prompt(self, query, answer, tool_results):
        # self reflection - check answer quality
        formatted = self.format_tool_results(tool_results)
        return f"""check this answer for issues.

question: {query}

answer: {answer}

sources:
{formatted}

check:
1. does it answer the question?
2. are claims backed by sources?
3. anything made up?

respond:
ISSUES: <problems or "none">
VERDICT: <PASS or FAIL>"""
