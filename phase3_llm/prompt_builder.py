class PromptBuilder:
    def __init__(self):
        # generic system prompt - llm discovers corpus through tools
        self.sys_prompt = """you are a research assistant that answers questions about academic papers.
use only the tool results provided to answer questions.
cite paper ids when possible.
be concise and factual.
if info isn't in the tool results, say so."""

        # tool descriptions - what they do, not what data exists
        self.tool_catalog = [
            {"name": "get_all_datasets", "desc": "list datasets mentioned in papers with usage counts"},
            {"name": "get_all_vision_models", "desc": "list vision encoders used across papers"},
            {"name": "get_training_setups", "desc": "get training configs from papers"},
            {"name": "get_all_hardware", "desc": "list hardware and sensors mentioned in papers"},
            {"name": "semantic_search", "desc": "search paper content by meaning"},
            {"name": "search_arxiv", "desc": "search arxiv for external papers not in the corpus"},
        ]

    def get_meta_policy(self, query):
        q = query.lower()

        if any(w in q for w in ['compare', 'vs', 'difference', 'versus']):
            return "comparison: list similarities and differences clearly"
        elif any(w in q for w in ['list', 'all', 'what are', 'which', 'compile']):
            return "list: use bullet points, include paper citations"
        elif any(w in q for w in ['how', 'why', 'explain', 'what is']):
            return "explain: be clear and cite sources"
        elif any(w in q for w in ['summarize', 'overview', 'brief']):
            return "summarize: keep it short, key points only"
        else:
            return "default: answer directly with citations"

    def build_decomposition_prompt(self, query):
        return f"""break this question into exactly 2 simpler sub-questions.

query: {query}

rules:
- output EXACTLY 2 questions, no more
- each must end with ?
- first question gathers basic info
- second question builds on the first
- do NOT include answers, only questions

format:
STEP1: <first question>
STEP2: <follow-up question>

example:
query: what datasets are popular and which papers use CLIP?
STEP1: what are the most common datasets?
STEP2: which papers use the CLIP vision model?

your response:"""

    def build_substep_prompt(self, step_query, prev_answers, tool_results):
        formatted = self.format_tool_results(tool_results)
        prev = ""
        if prev_answers:
            prev = "\ncontext from previous steps:\n"
            for i, ans in enumerate(prev_answers, 1):
                prev += f"- {ans}\n"

        return f"""{self.sys_prompt}
{prev}
question: {step_query}

data:
{formatted}

answer in 2-3 sentences, cite paper ids:"""

    def build_synthesis_prompt(self, original_query, step_answers):
        steps = "\n".join([f"- {ans}" for ans in step_answers])
        return f"""{self.sys_prompt}

original question: {original_query}

research findings:
{steps}

write a clear, complete answer combining the findings above.
keep it concise (3-5 sentences), cite papers when relevant:"""

    def build_planning_prompt(self, query):
        tools = "\n".join([f"- {t['name']}: {t['desc']}" for t in self.tool_catalog])
        return f"""pick tools to answer this query.

tools:
{tools}

query: {query}

rules:
- pick 1-4 tools (no duplicates)
- use semantic_search for specific topics
- use get_all_* tools for listing/counting
- use search_arxiv only for external papers

respond exactly:
TOOLS: tool1, tool2

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
            elif 'papers' in res:
                out += self._fmt_arxiv(res['papers'])
            else:
                out += str(res)[:500]
        return out

    def _fmt_datasets(self, datasets):
        txt = ""
        for d in datasets[:10]:
            name = d.get('name') or d.get('dataset_name', '?')
            cnt = d.get('paper_count', 0)
            txt += f"- {name}: {cnt} papers\n"
        return txt

    def _fmt_models(self, models):
        txt = ""
        for m in models[:10]:
            name = m.get('model_name') or m.get('vision_encoder', '?')
            cnt = m.get('paper_count', 0)
            txt += f"- {name}: {cnt} papers\n"
        return txt

    def _fmt_training(self, setups):
        txt = ""
        for s in setups[:5]:
            paper = s.get('paper_id', '?')
            parts = []
            if s.get('optimizer'): parts.append(f"opt={s['optimizer']}")
            if s.get('learning_rate'): parts.append(f"lr={s['learning_rate']}")
            if s.get('batch_size'): parts.append(f"batch={s['batch_size']}")
            if parts:
                txt += f"- [{paper}]: {', '.join(parts)}\n"
        if not txt:
            txt = "(no training configs available)\n"
        return txt

    def _fmt_hardware(self, hw):
        txt = ""
        for h in hw[:10]:
            name = h.get('name') or h.get('hardware_name', '?')
            htype = h.get('type', '')
            txt += f"- {name} ({htype})\n"
        if not txt:
            txt = "(limited hardware data)\n"
        return txt

    def _fmt_search(self, results):
        txt = ""
        for i, r in enumerate(results[:5], 1):
            paper = r.get('paper_id', '?')
            content = r.get('text', r.get('content', ''))[:250]
            txt += f"\n[{paper}]: {content}...\n"
        return txt

    def _fmt_arxiv(self, papers):
        txt = "(external arxiv results)\n"
        for p in papers[:3]:
            title = p.get('title', '?')[:60]
            txt += f"- {title}...\n"
        return txt

    def build_prompt(self, query, tool_results):
        formatted = self.format_tool_results(tool_results)
        policy = self.get_meta_policy(query)

        return f"""{self.sys_prompt}

response style: {policy}

question: {query}

data from tools:
{formatted}

answer the question using only the data above.
cite paper ids in brackets like [rt1] or [openvla].
if data is missing, say what's not available.

answer:"""

    def build_reflection_prompt(self, query, answer, tool_results):
        formatted = self.format_tool_results(tool_results)
        return f"""check if this answer is accurate.

question: {query}

answer given: {answer}

source data:
{formatted}

check:
1. does it answer the question?
2. are claims supported by the source data?
3. any hallucinated facts not in sources?

respond exactly:
ISSUES: <list problems, or "none">
VERDICT: PASS or FAIL"""
