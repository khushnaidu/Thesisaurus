class QueryPlanner:
    """figures out which tools to use based on keywords"""

    def __init__(self):
        # keyword patterns for routing queries
        self.patterns = {
            'datasets': ['common datasets', 'all datasets', 'list datasets'],
            'vision_models': ['vision models', 'visual models', 'what models'],
            'training': ['training setup', 'hyperparameter', 'optimizer', 'learning rate', 'batch size'],
            'hardware': ['what hardware', 'what robots', 'robot platforms'],
            'arxiv': ['arxiv', 'look up', 'web', 'find papers', 'search for papers', 'similar papers', 'recent papers'],
            'semantic': ['how does', 'explain', 'tell me about', 'what is', 'compare'],
        }

    def analyze_query(self, query):
        q = query.lower()
        matches = []
        for cat, keywords in self.patterns.items():
            for kw in keywords:
                if kw in q:
                    matches.append(cat)
                    break
        return matches

    def plan(self, query):
        cats = self.analyze_query(query)
        plan = {'query': query, 'categories': cats, 'tools': []}

        # check web search first
        if 'arxiv' in cats:
            plan['tools'].append({'name': 'search_arxiv', 'params': {'query': query, 'max_results': 5}})
            return plan

        # then database stuff
        if 'datasets' in cats:
            plan['tools'].append({'name': 'get_all_datasets', 'params': {}})
        if 'vision_models' in cats:
            plan['tools'].append({'name': 'get_all_vision_models', 'params': {}})
        if 'training' in cats:
            plan['tools'].append({'name': 'get_training_setups', 'params': {}})
        if 'hardware' in cats:
            plan['tools'].append({'name': 'get_all_hardware', 'params': {}})

        # fallback to semantic search
        if 'semantic' in cats or len(plan['tools']) == 0:
            plan['tools'].append({'name': 'semantic_search', 'params': {'query': query, 'top_k': 5}})

        return plan

    def execute_plan(self, plan, registry):
        results = {}
        for tool in plan['tools']:
            name = tool['name']
            params = tool['params']
            results[name] = registry.call_tool(name, **params)
        return results
