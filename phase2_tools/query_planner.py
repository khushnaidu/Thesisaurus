class QueryPlanner:
    """Analyzes queries and decides which tools to call"""
    
    def __init__(self):
        # Keyword patterns for routing
        self.patterns = {
            'datasets': ['common datasets', 'all datasets', 'list datasets'],
            'vision_models': ['vision models', 'visual models', 'what models'],
            'training': ['training setup', 'hyperparameter', 'optimizer', 'learning rate', 'batch size'],
            'hardware': ['what hardware', 'what robots', 'robot platforms'],
            'arxiv': ['arxiv', 'look up', 'web', 'find papers', 'search for papers', 'similar papers', 'recent papers'],
            'semantic': ['how does', 'explain', 'tell me about', 'what is', 'compare'],
        }
    
    def analyze_query(self, query):
        """
        Analyze query and return what type it is
        Returns: list of matched categories
        """
        query_lower = query.lower()
        matches = []
        
        for category, keywords in self.patterns.items():
            for keyword in keywords:
                if keyword in query_lower:
                    matches.append(category)
                    break
        
        return matches
    
    def plan(self, query):
        """
        Create execution plan for a query
        Returns dict with tools to call and their params
        """
        categories = self.analyze_query(query)
        plan = {
            'query': query,
            'categories': categories,
            'tools': []
        }
        
        # Priority 1: Check for web/arXiv search first
        if 'arxiv' in categories:
            plan['tools'].append({
                'name': 'search_arxiv',
                'params': {'query': query, 'max_results': 5}
            })
            return plan
        
        # Priority 2: Database queries (only if explicitly asked)
        if 'datasets' in categories:
            plan['tools'].append({
                'name': 'get_all_datasets',
                'params': {}
            })
        
        if 'vision_models' in categories:
            plan['tools'].append({
                'name': 'get_all_vision_models',
                'params': {}
            })
        
        if 'training' in categories:
            plan['tools'].append({
                'name': 'get_training_setups',
                'params': {}
            })
        
        if 'hardware' in categories:
            plan['tools'].append({
                'name': 'get_all_hardware',
                'params': {}
            })
        
        # Priority 3: Semantic search (default for everything else)
        if 'semantic' in categories or len(plan['tools']) == 0:
            plan['tools'].append({
                'name': 'semantic_search',
                'params': {'query': query, 'top_k': 5}
            })
        
        return plan
    
    def execute_plan(self, plan, tool_registry):
        """Execute the plan using tool registry"""
        results = {}
        
        for tool_spec in plan['tools']:
            tool_name = tool_spec['name']
            params = tool_spec['params']
            
            result = tool_registry.call_tool(tool_name, **params)
            results[tool_name] = result
        
        return results

